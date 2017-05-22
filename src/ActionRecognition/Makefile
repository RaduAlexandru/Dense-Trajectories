TOPDIR = ../

include ../definitions.make

OBJECTS = Example.o \
          KMeans.o \
          Gmm.o \
          Stip.o \
          FeatureQuantizer.o \
          Svm.o \
					Dense.o \
					Track.o \
					Histogram.o

OBJ = $(patsubst %, objects/%, $(OBJECTS))

LIB = ../Core/libCore.a ../Math/libMath.a ../Features/libFeatures.a

.PHONY: all prepare clean Application libActionRecognition.a

all: prepare $(OBJ) libActionRecognition.a Application

prepare:
	@mkdir -p objects

$(OBJ): objects/%.o : %.cc %.hh
	$(CC) -std=c++11 $(COPTS) -c $< -lcppunit -o $@

libActionDetector.a:
	$(MAKELIB) $(ARFLAGS) $@ $(OBJ)

Application: $(OBJ)
	$(CC) $(COPTS) $@.cc $(OBJ) -Wl,--start-group $(LIB) -Wl,--end-group $(CLIB) -o action-recognizer

clean:
	rm -rf objects/ libActionDetection.a action-recognizer
