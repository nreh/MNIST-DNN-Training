build:
	g++ src/main.cpp -O3 -Wno-sizeof-array-argument

debug:
	g++ src/main.cpp -g

clean:
	rm -f a.out