OUT_FOLDER := bin

all: $(OUT_FOLDER)/cnn $(OUT_FOLDER)/fc $(OUT_FOLDER)/relu $(OUT_FOLDER)/max_pool_2d $(OUT_FOLDER)/batch_norm_2d $(OUT_FOLDER)/conv_relu $(OUT_FOLDER)/conv_batch_relu $(OUT_FOLDER)/conv_batch_relu_max $(OUT_FOLDER)/full_nn

$(OUT_FOLDER)/cnn: tb/test_cnn.cpp tb/utils.cpp tb/utils.h src/cnn.cpp src/cnn.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/fc: tb/test_fc.cpp tb/utils.cpp tb/utils.h src/fc.cpp src/fc.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/relu: tb/test_relu.cpp tb/utils.cpp tb/utils.h src/relu.cpp src/relu.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/max_pool_2d: tb/test_max_pool_2d.cpp tb/utils.cpp tb/utils.h src/max_pool_2d.cpp src/max_pool_2d.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/batch_norm_2d: tb/test_batch_norm_2d.cpp tb/utils.cpp tb/utils.h src/batch_norm_2d.cpp src/batch_norm_2d.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/conv_relu: tb/test_conv_relu.cpp tb/utils.cpp tb/utils.h src/cnn.cpp src/cnn.h src/relu.cpp src/relu.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/conv_batch_relu: tb/test_conv_batch_relu.cpp tb/utils.cpp tb/utils.h src/cnn.cpp src/cnn.h src/relu.cpp src/relu.h src/batch_norm_2d.cpp src/batch_norm_2d.h src/conv_batch_relu.cpp src/conv_batch_relu.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/conv_batch_relu_max: tb/test_conv_batch_relu_max.cpp tb/utils.cpp tb/utils.h src/cnn.cpp src/cnn.h src/relu.cpp src/relu.h src/batch_norm_2d.cpp src/batch_norm_2d.h src/max_pool_2d.cpp src/max_pool_2d.h src/conv_batch_relu_max.cpp src/conv_batch_relu_max.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/full_nn: tb/test_full_nn.cpp tb/utils.cpp tb/utils.h src/cnn.cpp src/cnn.h src/relu.cpp src/relu.h src/batch_norm_2d.cpp src/batch_norm_2d.h src/max_pool_2d.cpp src/max_pool_2d.h src/conv_batch_relu.h src/conv_batch_relu.cpp src/conv_batch_relu_max.cpp src/conv_batch_relu_max.h src/full_nn.cpp src/full_nn.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm $(OUT_FOLDER)/*
