
target = libimgproc.a
objects = fastblur.o downsample.o utils.o
prefix = /usr/local
libdir = $(prefix)/lib

CC = g++
CFLAGS = -O3 -std=c++11 -mavx2
AR = ar
STRIP = strip

all: $(objects)
	$(AR) rcs $(target) $^

$(objects): %.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(target) $(objects)

install:
	cp $(target) $(libdir)

uninstall:
	rm -f $(libdir)/$(target)

