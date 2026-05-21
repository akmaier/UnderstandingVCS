# Build the frame-level trace-dump tool against a pre-built libxitari.a.
#
# Expects the xitari working copy to be at ../xitari with `libxitari.a`
# already built (run `cmake . && make` inside xitari/ first).
#
# Targets:
#   make           -> build trace_dump
#   make clean     -> remove trace_dump and any .o
#
# See README.md in this folder for usage and limitations.

XITARI_DIR ?= ../xitari

CXX       ?= c++
CXXFLAGS  ?= -std=c++11 -O2 -Wall -Wextra
INCLUDES   = -I$(XITARI_DIR) -I$(XITARI_DIR)/emucore -I$(XITARI_DIR)/emucore/m6502/src \
             -I$(XITARI_DIR)/emucore/m6502/src/bspf/src -I$(XITARI_DIR)/common \
             -I$(XITARI_DIR)/controllers -I$(XITARI_DIR)/environment -I$(XITARI_DIR)/games \
             -I$(XITARI_DIR)/os_dependent
LIBS       = $(XITARI_DIR)/libxitari.a -lz

BIN        = trace_dump

all: $(BIN)

$(BIN): trace_dump.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(BIN) *.o

.PHONY: all clean
