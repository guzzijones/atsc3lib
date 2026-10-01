CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra
PKG_CONFIG ?= pkg-config

SOAPY_CFLAGS := $(shell $(PKG_CONFIG) --cflags SoapySDR 2>/dev/null)
SOAPY_LIBS := $(shell $(PKG_CONFIG) --libs SoapySDR 2>/dev/null)

TOOLS_BIN = tools/soapy_capture

.PHONY: all soapy_capture clean

all: soapy_capture

soapy_capture: $(TOOLS_BIN)

$(TOOLS_BIN): tools/soapy_capture.c
	$(CC) $(CFLAGS) -o $@ $< $(SOAPY_CFLAGS) $(SOAPY_LIBS)

clean:
	rm -f $(TOOLS_BIN)
