objects = main.o game.o

all: $(objects)
	nvcc -arch=sm_30 $(objects) -o app

%.o: %.cpp
	nvcc -x cu -arch=sm_30 -I. -dc $< -o $@

%.o: src/%.cpp
	nvcc -x cu -arch=sm_30 -I. -dc $< -o $@

clean:
	rm -f *.o app.*
