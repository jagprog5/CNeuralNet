BUILDDIR = build
SOURCEDIR = src
HEADERDIR = headers

SOURCES = $(wildcard $(SOURCEDIR)/*.c)
OBJECTS = $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.o,$(SOURCES))

CFLAGS = -lm -lncurses
EXECUTABLE = $(BUILDDIR)/cneuralnet


.PHONY: all clean run run_visual run_reduced

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) -I$(HEADERDIR) $^ -o $@ $(CFLAGS)

$(BUILDDIR)/%.o: $(SOURCEDIR)/%.c
	$(CC) -I$(HEADERDIR) -c $< -o $@ $(CFLAGS)

clean:
	rm -vf $(EXECUTABLE) $(OBJECTS)

run: all
	$(EXECUTABLE)

# default is visual
run_visual: all
	./$(EXECUTABLE) 1

run_reduced: all
	./$(EXECUTABLE) 0

run_progression: all
	./$(EXECUTABLE) 2
