objects = main.obj game.obj

all: $(objects)
	nvcc -arch=sm_75 $(objects) -o game

%.obj: %.cpp
	nvcc -x cu -arch=sm_75 -I. -dc $< -o $@

%.obj: src/%.cpp
	nvcc -x cu -arch=sm_75 -I. -dc $< -o $@

clean:
	rm -f *.obj game.*

rebuild: clean all

run: 
	game

start: all run
