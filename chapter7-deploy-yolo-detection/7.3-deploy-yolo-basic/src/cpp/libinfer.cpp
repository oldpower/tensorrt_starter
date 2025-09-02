// libinfer.cpp
#include "trt_model.hpp"
#include "trt_logger.hpp"
#include "trt_worker.hpp"
#include "utils.hpp"

#include <mutex>
#include <memory>
#include <string>

using namespace std;

namespace {
    shared_ptr<thread::Worker> g_worker;
    mutex g_mutex;
}

extern "C" {

    // 初始化模型（只调用一次）
    void init_model(const char* onnx_path, int level, int device, int precision) {
        auto logger_level = static_cast<logger::Level>(level);
        auto model_device = static_cast<model::device>(device);
        auto model_precision = static_cast<model::precision>(precision);

        model::Params params;
        params.img = {640, 640, 3};
        params.task = model::task_type::DETECTION;
        params.dev = model_device;
        params.prec = model_precision;

        lock_guard<mutex> lock(g_mutex);
        if (!g_worker) {
            g_worker = thread::create_worker(onnx_path, logger_level, params);
        }
    }

    // 执行推理
    void run_inference(const char* image_path) {
        if (!g_worker) {
            // 模型未初始化
            return;
        }

        lock_guard<mutex> lock(g_mutex);
        g_worker->inference(string(image_path));
    }

    // 带返回值的推理（可选）
    const char* run_inference_with_result(const char* image_path) {
        if (!g_worker) {
            return "";
        }

        lock_guard<mutex> lock(g_mutex);
        g_worker->inference(string(image_path));

        // 示例返回值，你可以替换成实际推理结果
        static string result = "inference done";
        return result.c_str();
    }

} // extern "C"
