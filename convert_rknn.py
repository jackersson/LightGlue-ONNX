from rknn.api import RKNN
import fire

def convert(filename: str):
    
    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(target_platform="rk3588")
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=filename)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn("test.rknn")
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()

if __name__ == '__main__':
    fire.Fire(convert)
    