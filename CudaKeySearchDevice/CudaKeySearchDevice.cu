#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "KeySearchTypes.h"
#include "CudaKeySearchDevice.h"
#include "ptx.cuh"
#include "secp256k1.cuh"

#include "sha256.cuh"
#include "ripemd160.cuh"

#include "secp256k1.h"

#include "CudaHashLookup.cuh"
#include "CudaAtomicList.cuh"
#include "CudaDeviceKeys.cuh"

__constant__ unsigned int _INC_X[8];

__constant__ unsigned int _INC_Y[8];

__constant__ unsigned int *_CHAIN[1];

__constant__ unsigned int _TARGET_PUBKEY[8] = { 0xb4a72e4a, 0xaa69ba04, 0xb80c6891, 0xdf01f50d, 0x191a65ec, 0xcc61e4e9, 0x862d1e42, 0x1ce815b3 };
//__constant__ unsigned int _TARGET_PUBKEY[8] = { 0xdc6c9273, 0x4f925f9f, 0x88607afa, 0x26184554, 0x0c0768ab, 0x20dd5bc8, 0xcf9b43aa, 0x6bce911c };

static unsigned int *_chainBufferPtr = NULL;


__device__ void doRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
{
    const unsigned int iv[5] = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0
    };

    for(int i = 0; i < 5; i++) {
        hOut[i] = endian(hIn[i] + iv[(i + 1) % 5]);
    }
}


/**
 * Allocates device memory for storing the multiplication chain used in
 the batch inversion operation
 */
cudaError_t allocateChainBuf(unsigned int count)
{
    cudaError_t err = cudaMalloc(&_chainBufferPtr, count * sizeof(unsigned int) * 8);

    if(err) {
        return err;
    }

    err = cudaMemcpyToSymbol(_CHAIN, &_chainBufferPtr, sizeof(unsigned int *));
    if(err) {
        cudaFree(_chainBufferPtr);
    }

    return err;
}

void cleanupChainBuf()
{
    if(_chainBufferPtr != NULL) {
        cudaFree(_chainBufferPtr);
        _chainBufferPtr = NULL;
    }
}

/**
 *Sets the EC point which all points will be incremented by
 */
cudaError_t setIncrementorPoint(const secp256k1::uint256 &x, const secp256k1::uint256 &y)
{
    unsigned int xWords[8];
    unsigned int yWords[8];

    x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
    y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

    cudaError_t err = cudaMemcpyToSymbol(_INC_X, xWords, sizeof(unsigned int) * 8);
    if(err) {
        return err;
    }

    return cudaMemcpyToSymbol(_INC_Y, yWords, sizeof(unsigned int) * 8);
}



__device__ void hashPublicKey(const unsigned int *x, const unsigned int *y, unsigned int *digestOut)
{
    unsigned int hash[8];

    sha256PublicKey(x, y, hash);

    // Swap to little-endian
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}

__device__ void hashPublicKeyCompressed(const unsigned int *x, unsigned int yParity, unsigned int *digestOut)
{
    unsigned int hash[8];

    sha256PublicKeyCompressed(x, yParity, hash);

    // Swap to little-endian
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}


__device__ void setResultFound(int idx, bool findAddress, bool compressed, unsigned int x[8], unsigned int y[8], unsigned int digest[5])
{
    CudaDeviceResult r;

    r.block = blockIdx.x;
    r.thread = threadIdx.x;
    r.idx = idx;
    r.compressed = compressed;

    for(int i = 0; i < 8; i++) {
        r.x[i] = x[i];
        r.y[i] = y[i];
    }

    doRMD160FinalRound(digest, r.digest);

    atomicListAdd(&r, sizeof(r));
}

__device__ void doIteration(int pointsPerThread, int compression, bool findAddress)
{
    unsigned int *chain = _CHAIN[0];
    unsigned int *xPtr = ec::getXPtr();
    unsigned int *yPtr = ec::getYPtr();

    // Multiply together all (_Gx - x) and then invert
    unsigned int inverse[8] = {0,0,0,0,0,0,0,1};
    for(int i = 0; i < pointsPerThread; i++) {
        unsigned int x[8];

        readInt(xPtr, i, x);

        if (findAddress) {
            unsigned int digest[5];
            if (compression == PointCompressionType::UNCOMPRESSED || compression == PointCompressionType::BOTH) {
                unsigned int y[8];
                readInt(yPtr, i, y);

                hashPublicKey(x, y, digest);

                if (checkHash(digest)) {
                    setResultFound(i, true, false, x, y, digest);
                }
            }

            if (compression == PointCompressionType::COMPRESSED || compression == PointCompressionType::BOTH) {
                hashPublicKeyCompressed(x, readIntLSW(yPtr, i), digest);

                if (checkHash(digest)) {
                    unsigned int y[8];
                    readInt(yPtr, i, y);
                    setResultFound(i, true, true, x, y, digest);
                }
            }
        }
        else {

            bool equal = true;
            for (int i = 0; i < 8; i++) {
                equal &= (x[i] == _TARGET_PUBKEY[i]);
            }
            // todo: also check sign of y coordinate, whatever lazy
            if (equal) {
                unsigned int y[8];
                unsigned int digest[5];
                readInt(yPtr, i, y);
                hashPublicKey(x, y, digest);
                setResultFound(i, false, false, x, y, digest);
            }
        }

        beginBatchAdd(_INC_X, x, chain, i, i, inverse);
    }

    doBatchInverse(inverse);

    for(int i = pointsPerThread - 1; i >= 0; i--) {

        unsigned int newX[8];
        unsigned int newY[8];

        completeBatchAdd(_INC_X, _INC_Y, xPtr, yPtr, i, i, chain, inverse, newX, newY);

        writeInt(xPtr, i, newX);
        writeInt(yPtr, i, newY);
    }
}

__device__ void doIterationWithDouble(int pointsPerThread, int compression, bool findAddress)
{
    unsigned int *chain = _CHAIN[0];
    unsigned int *xPtr = ec::getXPtr();
    unsigned int *yPtr = ec::getYPtr();

    // Multiply together all (_Gx - x) and then invert
    unsigned int inverse[8] = {0,0,0,0,0,0,0,1};
    for(int i = 0; i < pointsPerThread; i++) {
        unsigned int x[8];


        readInt(xPtr, i, x);

        if (findAddress) {
            unsigned int digest[5];
            // uncompressed
            if (compression == PointCompressionType::UNCOMPRESSED || compression == PointCompressionType::BOTH) {
                unsigned int y[8];
                readInt(yPtr, i, y);
                hashPublicKey(x, y, digest);

                if (checkHash(digest)) {
                    setResultFound(i, true, false, x, y, digest);
                }
            }

            // compressed
            if (compression == PointCompressionType::COMPRESSED || compression == PointCompressionType::BOTH) {

                hashPublicKeyCompressed(x, readIntLSW(yPtr, i), digest);

                if (checkHash(digest)) {

                    unsigned int y[8];
                    readInt(yPtr, i, y);

                    setResultFound(i, true, true, x, y, digest);
                }
            }
        }
        else {

            bool equal = true;
            for (int j = 0; j < 8; j++) {
                equal &= (x[j] == _TARGET_PUBKEY[j]);
            }
            // todo: also check sign of y coordinate, whatever lazy
            if (equal) {
                unsigned int y[8];
                unsigned int digest[5];
                readInt(yPtr, i, y);
                hashPublicKey(x, y, digest);
                setResultFound(i, false, false, x, y, digest);
            }
        }

        beginBatchAddWithDouble(_INC_X, _INC_Y, xPtr, chain, i, i, inverse);
    }

    doBatchInverse(inverse);

    for(int i = pointsPerThread - 1; i >= 0; i--) {

        unsigned int newX[8];
        unsigned int newY[8];

        completeBatchAddWithDouble(_INC_X, _INC_Y, xPtr, yPtr, i, i, chain, inverse, newX, newY);

        writeInt(xPtr, i, newX);
        writeInt(yPtr, i, newY);
    }
}

/**
* Performs a single iteration
*/
__global__ void keyFinderKernel(int points, int compression)
{
    bool findAddress = false;
    doIteration(points, compression, findAddress);
}

__global__ void keyFinderKernelWithDouble(int points, int compression)
{
    bool findAddress = false;
    doIterationWithDouble(points, compression, findAddress);
}