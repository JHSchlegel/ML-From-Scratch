# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -Wall -I.

# Directories
SRCDIR = .
BUILDDIR = build

# Files
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(SOURCES))
TARGET = $(BUILDDIR)/kmeans_project

# Default target
all: $(TARGET)

# Link
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@

# Compile
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(BUILDDIR)/*

# Phony targets
.PHONY: all clean
