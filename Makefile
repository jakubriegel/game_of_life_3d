objects = main.obj game.obj

all: $(objects)
	nvcc -arch=sm_30 $(objects) -o game

%.obj: %.cpp
	nvcc -x cu -arch=sm_30 -I. -dc $< -o $@

%.obj: src/%.cpp
	nvcc -x cu -arch=sm_30 -I. -dc $< -o $@

clean:
	rm -f *.obj game.*
