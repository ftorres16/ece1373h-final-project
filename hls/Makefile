OUT_FOLDER := bin

cnn: tb/test_cnn.cpp tb/utils.cpp tb/utils.h src/cnn.cpp src/cnn.h
	$(CXX) $(CFLAGS) tb/test_cnn.cpp tb/utils.cpp tb/utils.h src/cnn.cpp src/cnn.h $(LDFLAGS) -o $(OUT_FOLDER)/$@

fc: tb/test_fc.cpp tb/utils.cpp tb/utils.h src/fc.cpp src/fc.h
	$(CXX) $(CFLAGS) tb/test_fc.cpp tb/utils.cpp tb/utils.h src/fc.cpp src/fc.h $(LDFLAGS) -o $(OUT_FOLDER)/$@

relu: tb/test_relu.cpp tb/utils.cpp tb/utils.h src/relu.cpp src/relu.h
	$(CXX) $(CFLAGS) tb/test_relu.cpp tb/utils.h tb/utils.cpp src/relu.cpp src/relu.h $(LDFLAGS) -o $(OUT_FOLDER)/$@

max_pool_2d: tb/test_max_pool_2d.cpp tb/utils.cpp tb/utils.h src/max_pool_2d.cpp src/max_pool_2d.h
	$(CXX) $(CFLAGS) tb/test_max_pool_2d.cpp tb/utils.h tb/utils.cpp src/max_pool_2d.cpp src/max_pool_2d.h $(LDFLAGS) -o $(OUT_FOLDER)/$@

batch_norm_2d: tb/test_batch_norm_2d.cpp tb/utils.cpp tb/utils.h src/batch_norm_2d.cpp src/batch_norm_2d.h
	$(CXX) $(CFLAGS) tb/test_batch_norm_2d.cpp tb/utils.h tb/utils.cpp src/batch_norm_2d.cpp src/batch_norm_2d.h $(LDFLAGS) -o $(OUT_FOLDER)/$@

conv_relu: tb/test_conv_relu.cpp tb/utils.cpp tb/utils.h src/cnn.cpp src/cnn.h src/relu.cpp src/relu.h
	$(CXX) $(CFLAGS) tb/test_conv_relu.cpp tb/utils.cpp tb/utils.h src/cnn.cpp src/cnn.h src/relu.cpp src/relu.h $(LDFLAGS) -o $(OUT_FOLDER)/$@

all: cnn fc relu max_pool_2d batch_norm_2d conv_relu

clean:
	rm $(OUT_FOLDER)/*
