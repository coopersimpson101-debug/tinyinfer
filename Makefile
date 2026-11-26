# tinyinfer - macOS / Linux build.  (Windows: use `python build.py`.)
#
#   make            CLI, quantiser, C tests, kernel bench, shared library (CPU)
#   make metal      macOS only: CLI with the Metal backend -> build/tinyinfer-metal
#   make test       C unit tests + Python test-suite
#   make clean

CC      ?= cc
BUILD   ?= build
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

CFLAGS  ?= -O3
CFLAGS  += -std=c11 -Wall -Wextra -Wno-unused-parameter
LDLIBS  += -lm

ifeq ($(UNAME_S),Darwin)
  SHLIB := libtinyinfer.dylib
  ifeq ($(UNAME_M),arm64)
    ARCH ?= -mcpu=apple-m1          # NEON + SDOT (__ARM_FEATURE_DOTPROD)
  else
    ARCH ?= -march=native
  endif
else
  SHLIB := libtinyinfer.so
  ARCH ?= -march=native
  LDLIBS += -lpthread
endif
CFLAGS += $(ARCH)

LIB_SRC := src/platform.c src/quant.c src/kernels.c src/gguf.c src/tokenizer.c src/model.c src/sampler.c \
           src/kvcache.c src/speculative.c
HDRS    := $(wildcard src/*.h) include/tinyinfer.h

.PHONY: all metal test test-c test-py clean
all: $(BUILD)/tinyinfer $(BUILD)/tinyinfer-quantize $(BUILD)/test_core $(BUILD)/bench_kernels $(BUILD)/$(SHLIB)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/tinyinfer: $(LIB_SRC) src/main.c $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(LIB_SRC) src/main.c $(LDLIBS)

$(BUILD)/tinyinfer-quantize: src/platform.c src/quant.c src/gguf.c src/quantize_main.c $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ src/platform.c src/quant.c src/gguf.c src/quantize_main.c $(LDLIBS)

$(BUILD)/test_core: $(LIB_SRC) tests/test_core.c $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(LIB_SRC) tests/test_core.c $(LDLIBS)

$(BUILD)/bench_kernels: $(LIB_SRC) tools/bench_kernels.c $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(LIB_SRC) tools/bench_kernels.c $(LDLIBS)

$(BUILD)/$(SHLIB): $(LIB_SRC) $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(LIB_SRC) $(LDLIBS)

# ---- Metal (macOS) ---------------------------------------------------------
# The shader source is embedded in the binary (xxd -i from stdin emits just the
# byte list) and compiled at start-up with newLibraryWithSource, so only the
# Command Line Tools are needed, not the full Xcode `metal` toolchain.
$(BUILD)/kernels_metal.inc: metal/kernels.metal | $(BUILD)
	xxd -i < $< > $@

metal: $(BUILD)/tinyinfer-metal

$(BUILD)/tinyinfer-metal: $(LIB_SRC) src/main.c metal/ti_metal.m metal/ti_metal.h $(BUILD)/kernels_metal.inc $(HDRS)
ifneq ($(UNAME_S),Darwin)
	@echo "make metal requires macOS" && exit 1
endif
	$(CC) $(CFLAGS) -DTI_METAL -c $(LIB_SRC) src/main.c
	$(CC) -O2 -fno-objc-arc -DTI_METAL -I$(BUILD) -c metal/ti_metal.m -o ti_metal.o
	$(CC) -o $@ *.o -framework Metal -framework Foundation $(LDLIBS)
	rm -f *.o

# ---- tests -----------------------------------------------------------------
test: test-c test-py

test-c: $(BUILD)/test_core
	$(BUILD)/test_core

test-py: $(BUILD)/$(SHLIB) $(BUILD)/tinyinfer $(BUILD)/tinyinfer-quantize $(BUILD)/test_core
	TI_NO_AUTOBUILD=1 python3 -m pytest -q tests

clean:
	rm -rf $(BUILD) *.o
