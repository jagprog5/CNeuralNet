BUILDDIR = build
SOURCEDIR = src
HEADERDIR = headers

SOURCES = $(wildcard $(SOURCEDIR)/*.c)
OBJECTS = $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.o,$(SOURCES))

CFLAGS = -lm
EXECUTABLE = $(BUILDDIR)/cneuralnet


.PHONY: all clean run run_visual run_reduced

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) -I$(HEADERDIR) $^ -o $@

$(BUILDDIR)/%.o: $(SOURCEDIR)/%.c
	$(CC) $(CFLAGS) -I$(HEADERDIR) -c $< -o $@

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
