cnn:
	$(CXX) $(CFLAGS) tb/test_cnn.cpp src/cnn.cpp src/cnn.h $(LDFLAGS) -o $@

fc:
	$(CXX) $(CFLAGS) tb/test_fc.cpp tb/utils.cpp tb/utils.h src/fc.cpp src/fc.h $(LDFLAGS) -o $@

relu:
	$(CXX) $(CFLAGS) tb/test_relu.cpp src/relu.cpp src/relu.h $(LDFLAGS) -o $@

all: cnn fc relu

clean:
	rm cnn fc relu
