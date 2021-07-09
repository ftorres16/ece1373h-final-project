OUT_FOLDER := bin

vpath %.cpp tb src
vpath %.h tb src

.PHONY: all
all: $(OUT_FOLDER)/conv $(OUT_FOLDER)/fc $(OUT_FOLDER)/relu $(OUT_FOLDER)/max_pool_2d $(OUT_FOLDER)/batch_norm_2d $(OUT_FOLDER)/conv_relu $(OUT_FOLDER)/conv_batch_relu $(OUT_FOLDER)/conv_batch_relu_max $(OUT_FOLDER)/full_nn

$(OUT_FOLDER)/conv: test_conv.cpp utils.cpp utils.h conv.cpp conv.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/fc: test_fc.cpp utils.cpp utils.h fc.cpp fc.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/relu: test_relu.cpp utils.cpp utils.h relu.cpp relu.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/max_pool_2d: test_max_pool_2d.cpp utils.cpp utils.h max_pool_2d.cpp max_pool_2d.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/batch_norm_2d: test_batch_norm_2d.cpp utils.cpp utils.h batch_norm_2d.cpp batch_norm_2d.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/conv_relu: test_conv_relu.cpp utils.cpp utils.h conv.cpp conv.h relu.cpp relu.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/conv_batch_relu: test_conv_batch_relu.cpp utils.cpp utils.h conv.cpp conv.h relu.cpp relu.h batch_norm_2d.cpp batch_norm_2d.h conv_batch_relu.cpp conv_batch_relu.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/conv_batch_relu_max: test_conv_batch_relu_max.cpp utils.cpp utils.h conv.cpp conv.h relu.cpp relu.h batch_norm_2d.cpp batch_norm_2d.h max_pool_2d.cpp max_pool_2d.h conv_batch_relu_max.cpp conv_batch_relu_max.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/full_nn: test_full_nn.cpp utils.cpp utils.h conv.cpp conv.h relu.cpp relu.h batch_norm_2d.cpp batch_norm_2d.h max_pool_2d.cpp max_pool_2d.h conv_batch_relu.h conv_batch_relu.cpp conv_batch_relu_max.cpp conv_batch_relu_max.h full_nn.cpp full_nn.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

.PHONY: clean
clean:
	rm $(OUT_FOLDER)/*
