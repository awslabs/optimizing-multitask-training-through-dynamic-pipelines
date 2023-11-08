CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
OS := $(shell uname)
ifeq ($(OS),Darwin)
	CXXFLAGS += -undefined dynamic_lookup
endif

CPPFLAGS += $(shell python3 -m pybind11 --includes)
LIBNAME = dp_helper
LIBEXT = $(shell python3-config --extension-suffix)


default: $(LIBNAME)$(LIBEXT)

%$(LIBEXT): %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
