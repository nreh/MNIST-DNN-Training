build:
	g++ src/main.cpp -O3 -oa -Wno-sizeof-array-argument

debug:
	g++ src/main.cpp -oa -g

clean:
	rm -f a.out