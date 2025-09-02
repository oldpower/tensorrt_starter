CONFIG        := ../../config/Makefile.config
CONFIG_LOCAL  := ./config/Makefile.config

include $(CONFIG)
include $(CONFIG_LOCAL)

BUILD_PATH    := build
SRC_PATH      := src/cpp
INC_PATH      := include
CUDA_DIR      := /usr/local/cuda-$(CUDA_VER)

# 源文件定义
CXX_SRC       := $(wildcard $(SRC_PATH)/*.cpp)
KERNELS_SRC   := $(wildcard $(SRC_PATH)/*.cu)

# 排除 libinfer.cpp 用于可执行文件构建
APP_CXX_SRC   := $(filter-out $(SRC_PATH)/libinfer.cpp, $(CXX_SRC))
LIB_CXX_SRC   := $(filter-out $(SRC_PATH)/main.cpp, $(CXX_SRC))

# 对象文件
APP_OBJS      := $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(APP_CXX_SRC:.cpp=.cpp.o))
LIB_OBJS      := $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(LIB_CXX_SRC:.cpp=.cpp.o))
CUDA_OBJS     := $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(KERNELS_SRC:.cu=.cu.o))

APP_OBJS      += $(CUDA_OBJS)
LIB_OBJS      += $(CUDA_OBJS)

APP_MKS       := $(APP_OBJS:.o=.mk) $(LIB_OBJS:.o=.mk)

# 目标和依赖
APP           := main
LIB           := libinfer.so

APP_DEPS      := $(APP_CXX_SRC) $(KERNELS_SRC) $(wildcard $(SRC_PATH)/*.h)
LIB_DEPS      := $(LIB_CXX_SRC) $(KERNELS_SRC) $(wildcard $(SRC_PATH)/*.h)

# 工具和标志 (保持不变)
CUCC          := $(CUDA_DIR)/bin/nvcc
CXXFLAGS      := -std=c++11 -pthread -fPIC
CUDAFLAGS     := --shared -Xcompiler -fPIC

INCS          := -I $(CUDA_DIR)/include \
                 -I $(SRC_PATH) \
                 -I $(OPENCV_INSTALL_DIR) \
                 -I $(TENSORRT_INSTALL_DIR)/include \
                 -I $(CUDNN_INSTALL_DIR)/include \
                 -I $(INC_PATH)

LIBS          := -L "$(CUDA_DIR)/lib64" \
                 -L "$(TENSORRT_INSTALL_DIR)/lib" \
                 -L "$(CUDNN_INSTALL_DIR)/lib" \
                 -lcudart -lcublas -lcudnn \
                 -lnvinfer -lnvonnxparser \
                 -lnvinfer_plugin \
                 -lstdc++fs \
                 `pkg-config --libs opencv4`

# 调试和警告标志 (保持不变)
ifeq ($(DEBUG),1)
CUDAFLAGS     += -g -O0
CXXFLAGS      += -g -O0
else
CUDAFLAGS     += -O3
CXXFLAGS      += -O3
endif

ifeq ($(SHOW_WARNING),1)
CUDAFLAGS     += -Wall -Wunused-function -Wunused-variable -Wfatal-errors
CXXFLAGS      += -Wall -Wunused-function -Wunused-variable -Wfatal-errors
else
CUDAFLAGS     += -w
CXXFLAGS      += -w
endif

# bear命令 (保持不变)
ifeq (, $(shell which bear))
BEARCMD       :=
else
ifeq (bear 3.0.18, $(shell bear --version))
BEARCMD       := bear --output config/compile_commands.json --
else
BEARCMD       := bear -o config/compile_commands.json
endif
endif

# 目标定义
all: lib app

lib: $(LIB)

app: $(APP)

run: app
	@./bin/$(APP)

$(APP): $(APP_OBJS)
	@echo "Linking executable $@"
	@mkdir -p bin
	@$(CXX) $^ -o bin/$@ $(LIBS) $(INCS)

$(LIB): $(LIB_OBJS)
	@echo "Linking shared library $@"
	@mkdir -p bin
	@$(CXX) -shared $^ -o bin/$@ $(LIBS) $(INCS)

update: $(APP)
	@echo finished updating $<

show:
	@echo "Build path: $(BUILD_PATH)"
	@echo "App sources: $(APP_CXX_SRC)"
	@echo "Lib sources: $(LIB_CXX_SRC)"
	@echo "CUDA sources: $(KERNELS_SRC)"
	@echo "Include paths: $(INCS)"
	@echo "App objects: $(APP_OBJS)"
	@echo "Lib objects: $(LIB_OBJS)"

clean:
	rm -rf bin/$(APP) bin/$(LIB)
	rm -rf build
	rm -rf config/compile_commands.json
	rm -rf bin

# 编译规则 (保持不变)
$(BUILD_PATH)/%.cpp.o: $(SRC_PATH)/%.cpp
	@echo "Compiling CXX $<"
	@mkdir -p $(BUILD_PATH)
	@$(CXX) -o $@ -c $< $(CXXFLAGS) $(INCS)

$(BUILD_PATH)/%.cpp.mk: $(SRC_PATH)/%.cpp
	@echo "Generating dependencies for $<"
	@mkdir -p $(BUILD_PATH)
	@$(CXX) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(CXXFLAGS) $(INCS)

$(BUILD_PATH)/%.cu.o: $(SRC_PATH)/%.cu
	@echo "Compiling CUDA $<"
	@mkdir -p $(BUILD_PATH)
	@$(CUCC) -o $@ -c $< $(CUDAFLAGS) $(INCS)

$(BUILD_PATH)/%.cu.mk: $(SRC_PATH)/%.cu
	@echo "Generating CUDA dependencies for $<"
	@mkdir -p $(BUILD_PATH)
	@$(CUCC) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(CUDAFLAGS)

.PHONY: all lib app run update show clean
