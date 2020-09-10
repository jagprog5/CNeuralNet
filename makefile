BUILDDIR = build
SOURCEDIR = src
HEADERDIR = headers

SOURCES := $(wildcard $(SOURCEDIR)/*.c)
OBJECTS := $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.o,$(SOURCES))

CFLAGS = -Ofast -std=c99
LINKFLAGS = -lm -lncurses
EXECUTABLE := $(BUILDDIR)/cneuralnet

.PHONY: all clean run

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $^ -o $@ $(LINKFLAGS)

$(BUILDDIR)/%.o: $(SOURCEDIR)/%.c
	$(CC) -I$(HEADERDIR) -c $< -o $@ $(CFLAGS)

clean:
	rm -vf $(EXECUTABLE) $(OBJECTS)

run: all
	$(EXECUTABLE)
