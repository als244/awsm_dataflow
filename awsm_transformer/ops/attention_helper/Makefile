CC = gcc
DEV_CFLAGS = -g -fPIC
BUILD_CFLAGS = -O3 -fPIC
CFLAGS = ${DEV_CFLAGS}

CUDA_INCLUDE_DIR = /usr/local/cuda/include
CUDA_LIB_DIR = /usr/local/cuda/lib64/stubs
CUDA_LIB_DEPENDS = -lcuda


## Native OP

NATIVE_OP_PATH = src/native

## setting up variables for native op kernels

NVCC = nvcc
DEV_NVCC_FLAGS = -g -G -Xptxas=-v
BUILD_NVCC_FLAGS = -O4 --use_fast_math
PROF_NVCC_FLAGS = -O4 --use_fast_math --generate-line-info -Xptxas=-v
NVCC_FLAGS = ${PROF_NVCC_FLAGS}


## Attention Helper Library

## 0.) create flash3 lib as helper lib

FLASH3_DIR = flash3

### flash3 lib dir
FLASH3_LIB_DIR = ${FLASH3_DIR}/lib
### flash3 include dir
FLASH3_INCLUDE_DIR = ${FLASH3_DIR}/include
### flash3 lib
FLASH3_LIB = -lflash3


FLASH2_DIR = flash2
FLASH2_LIB_DIR = ${FLASH2_DIR}/lib
FLASH2_INCLUDE_DIR = ${FLASH2_DIR}/include
FLASH2_LIB = -lflash2

## ensure we make flash3 lib


## 1.) set up variables for attention helper lib

EXTERNAL_OP_ATTENTION_LIB_OBJS = objs/attention_helper.o
EXTERNAL_OP_ATTENTION_LIB_ALL_INCLUDES = -I${CUDA_INCLUDE_DIR} -I${FLASH3_INCLUDE_DIR} -I${FLASH2_INCLUDE_DIR} -Iinclude
EXTERNAL_OP_ATTENTION_LIB_OTHER_LIB_DEPENDS = -L${FLASH3_LIB_DIR} ${FLASH3_LIB} -Wl,-rpath,${FLASH3_LIB_DIR} -L${FLASH2_LIB_DIR} ${FLASH2_LIB} -Wl,-rpath,${FLASH2_LIB_DIR}


EXTERNAL_OP_ATTENTION_LIB = libattentionwrapper.so



## External Attention Lib
lib/${EXTERNAL_OP_ATTENTION_LIB}: build_flash2_lib build_flash3_lib ${EXTERNAL_OP_ATTENTION_LIB_OBJS}
	${CC} ${CFLAGS} ${EXTERNAL_OP_ATTENTION_LIB_ALL_INCLUDES} -shared ${EXTERNAL_OP_ATTENTION_LIB_OBJS} -o $@ ${EXTERNAL_OP_ATTENTION_LIB_OTHER_LIB_DEPENDS}


# Retry loops for flash libs because nvcc sometimes has errors...
build_flash3_lib:
	@for i in $$(seq 1 20); do \
        if ${MAKE} -C ${FLASH3_DIR}; then \
            echo "Successfully built flash3 lib."; \
            break; \
        fi; \
        if [ $$i -eq 20 ]; then \
            echo "Error: Failed to build flash3 lib after 20 attempts. Something is likely wrong..."; \
            exit 1; \
        fi; \
        echo "make -C ${FLASH3_DIR} failed. Retrying (attempt $$i of 20)..."; \
        sleep 10; \
    done

build_flash2_lib:
	@for i in $$(seq 1 20); do \
        if ${MAKE} -C ${FLASH2_DIR}; then \
            echo "Successfully built flash2 lib."; \
            break; \
        fi; \
        if [ $$i -eq 20 ]; then \
            echo "Error: Failed to build flash2 lib after 20 attempts. Something is likely wrong..."; \
            exit 1; \
        fi; \
        echo "make -C ${FLASH2_DIR} failed. Retrying (attempt $$i of 20)..."; \
        sleep 10; \
    done


objs/attention_helper.o: src/attention_helper.c
	${CC} ${CFLAGS} ${EXTERNAL_OP_ATTENTION_LIB_ALL_INCLUDES} -c $< -o $@


## don't wipe flash3 lib as this takes a while to build...
## can go into that directory and run clean if wanted...
clean:
	rm -f objs/*.o lib/${EXTERNAL_OP_ATTENTION_LIB}
