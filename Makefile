target:
	mkdir target

build: target
	nvcc -o target/simpletexture.out simpletexture.cu Timer.cpp --ptxas-options=-v

run: target/simpletexture.out
	./target/simpletexture.out

clean:
	rm -rf target
	