OUT_FOLDER := bin

vpath %.cpp tb tb/layers tb/spike_deeptector src src/layers src/spike_deeptector
vpath %.h tb src src/layers src/spike_deeptector

.PHONY: all
all: $(OUT_FOLDER)/conv $(OUT_FOLDER)/fc $(OUT_FOLDER)/relu $(OUT_FOLDER)/zero_mean $(OUT_FOLDER)/max_pool_2d $(OUT_FOLDER)/batch_norm_2d $(OUT_FOLDER)/conv_relu $(OUT_FOLDER)/conv_batch_relu $(OUT_FOLDER)/conv_batch_relu_max $(OUT_FOLDER)/spike_deeptector $(OUT_FOLDER)/bar $(OUT_FOLDER)/spike_deeptector_single_run $(OUT_FOLDER)/spike_deeptector_cumulative $(OUT_FOLDER)/spike_deeptector_main

$(OUT_FOLDER)/conv: test_conv.cpp utils.cpp utils.h conv.cpp conv.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/fc: test_fc.cpp utils.cpp utils.h fc.cpp fc.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/relu: test_relu.cpp utils.cpp utils.h relu.cpp relu.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/zero_mean: test_zero_mean.cpp utils.cpp utils.h zero_mean.cpp zero_mean.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/max_pool_2d: test_max_pool_2d.cpp utils.cpp utils.h max_pool_2d.cpp max_pool_2d.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/batch_norm_2d: test_batch_norm_2d.cpp utils.cpp utils.h batch_norm_2d.cpp batch_norm_2d.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/conv_relu: test_conv_relu.cpp utils.cpp utils.h conv_relu.cpp conv_relu.h conv.cpp conv.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/conv_batch_relu: test_conv_batch_relu.cpp utils.cpp utils.h conv.cpp conv.h relu.cpp relu.h batch_norm_2d.cpp batch_norm_2d.h conv_batch_relu.cpp conv_batch_relu.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/conv_batch_relu_max: test_conv_batch_relu_max.cpp utils.cpp utils.h conv.cpp conv.h relu.cpp relu.h batch_norm_2d.cpp batch_norm_2d.h max_pool_2d.cpp max_pool_2d.h conv_batch_relu.cpp conv_batch_relu.h conv_batch_relu_max.cpp conv_batch_relu_max.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/spike_deeptector: test_spike_deeptector.cpp utils.cpp utils.h conv.cpp conv.h conv_relu.cpp conv_relu.h batch_norm_2d.cpp batch_norm_2d.h max_pool_2d.cpp max_pool_2d.h conv_batch_relu.h conv_batch_relu.cpp conv_batch_relu_max.cpp conv_batch_relu_max.h zero_mean.h zero_mean.cpp spike_deeptector.cpp spike_deeptector.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/spike_deeptector_single_run: test_spike_deeptector_single_run.cpp utils.cpp utils.h conv.cpp conv.h conv_relu.cpp conv_relu.h batch_norm_2d.cpp batch_norm_2d.h max_pool_2d.cpp max_pool_2d.h conv_batch_relu.h conv_batch_relu.cpp conv_batch_relu_max.cpp conv_batch_relu_max.h zero_mean.h zero_mean.cpp spike_deeptector.cpp spike_deeptector.h	spike_deeptector_single_run.h spike_deeptector_single_run.cpp
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/spike_deeptector_cumulative: test_spike_deeptector_cumulative.cpp spike_deeptector_cumulative.cpp spike_deeptector_cumulative.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/spike_deeptector_main: test_spike_deeptector_main.cpp utils.cpp utils.h conv.cpp conv.h conv_relu.cpp conv_relu.h batch_norm_2d.cpp batch_norm_2d.h max_pool_2d.cpp max_pool_2d.h conv_batch_relu.h conv_batch_relu.cpp conv_batch_relu_max.cpp conv_batch_relu_max.h zero_mean.h zero_mean.cpp spike_deeptector.cpp spike_deeptector.h	spike_deeptector_single_run.h spike_deeptector_single_run.cpp spike_deeptector_cumulative.cpp spike_deeptector_cumulative.h spike_deeptector_main.cpp spike_deeptector_main.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT_FOLDER)/bar: test_bar.cpp utils.cpp utils.h conv.cpp conv.h conv_relu.cpp conv_relu.h batch_norm_2d.cpp batch_norm_2d.h max_pool_2d.cpp max_pool_2d.h conv_batch_relu.h conv_batch_relu.cpp conv_batch_relu_max.cpp conv_batch_relu_max.h zero_mean.h zero_mean.cpp bar.cpp bar.h
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

.PHONY: clean
clean:
	rm $(OUT_FOLDER)/*
