cnn: tb/test_cnn.cpp tb/utils.cpp tb/utils.h src/cnn.cpp src/cnn.h
	$(CXX) $(CFLAGS) tb/test_cnn.cpp tb/utils.cpp tb/utils.h src/cnn.cpp src/cnn.h $(LDFLAGS) -o $@

fc: tb/test_fc.cpp tb/utils.cpp tb/utils.h src/fc.cpp src/fc.h
	$(CXX) $(CFLAGS) tb/test_fc.cpp tb/utils.cpp tb/utils.h src/fc.cpp src/fc.h $(LDFLAGS) -o $@

relu: tb/test_relu.cpp tb/utils.cpp tb/utils.h src/relu.cpp src/relu.h
	$(CXX) $(CFLAGS) tb/test_relu.cpp tb/utils.h tb/utils.cpp src/relu.cpp src/relu.h $(LDFLAGS) -o $@

all: cnn fc relu

clean:
	rm cnn fc relu
