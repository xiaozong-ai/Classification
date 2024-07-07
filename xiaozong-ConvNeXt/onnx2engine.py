import os
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def get_engine(onnx_file_path, engine_file_path):
    if os.path.exists(engine_file_path):
        print("engine file already exists!!!")
        with open(engine_file_path, 'rb') as f, trt.Runtime(trt.Logger(TRT_LOGGER)) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    else:
        # create builder and network.
        builder = trt.Builder(TRT_LOGGER)
        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        # parse onnx.
        parser = trt.OnnxParser(network, logger=TRT_LOGGER)
        # with open(onnx_file_path, 'r', encoding='utf-8') as model:
        #     if not parser.parse(model.read()):
        #         print("Could not to parse the ONNX file!!!")
        if not parser.parse_from_file(onnx_file_path):
            raise RuntimeError(f'failed to load ONNX file: {onnx_file_path}')

        # create config.
        config = builder.create_builder_config()
        half = builder.platform_has_fast_fp16
        if half:
            config.set_flag(trt.BuilderFlag.FP16)
        builder.max_batch_size = 1
        config.max_workspace_size = 4 << 30     # Set the max_workspace_size is 4G.

        # create engine.
        engine = builder.build_engine(network, config)
        # save engine model.
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())


if __name__ == '__main__':
    onnx_model_path = "logs/yolov5m6.onnx"
    engine_model_path = "logs/best_model.engine"
    get_engine(onnx_model_path, engine_model_path)
