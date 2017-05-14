TOPDIR = ../

include ../definitions.make

OBJECTS = Random.o

OBJ = $(patsubst %, objects/%, $(OBJECTS))

LIB = ../Core/libCore.a

.PHONY: all prepare clean libMath.a

all: prepare $(OBJ) libMath.a

prepare:
	@mkdir -p objects

$(OBJ): objects/%.o : %.cc %.hh
	$(CC) $(COPTS) -c $< -lcppunit -o $@

libMath.a:
	$(MAKELIB) $(ARFLAGS) $@ $(OBJ)

clean:
	rm -rf objects/ libMath.a
