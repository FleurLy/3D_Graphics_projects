CC     = gcc
CFLAGS = -O3 -march=native -fPIC -shared -lm

TARGETS = box.so gaussian_downsample.so lanczos_kernel_vals.so lanczos_downsample.so med.so

all: $(TARGETS)

%.so: %.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
