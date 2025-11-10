from keras.models import load_model

_path = '/workspace/model/segL-1024/'
model_name = [
    'best_model_7200_m-acc_b-256_lr-002_rlr-80.h5','best_model_7213_m-val_acc_b-128.h5','best_model_7249_m-val_acc_b-128_lr-003.h5',
    'best_model_7204_m-val_acc.h5', 'best_model_7217_m-val_acc_b-256_lr-0005.h5','best_model_7258_m-val_acc.h5',
    'best_model_7210_m-acc.h5', 'best_model_7235_m-val_acc_b-128.h5','best_model_7285_m-val_loss_b-128_lr-003.h5',
    'best_model_7211.h5', 'best_model_7249_m-val_acc.h5','best_model_7345_m-val_acc_b-128_lr-004.h5'
]

for name in model_name:
    model_path = _path + name
    model = load_model(model_path)
    summary_path = f'/workspace/model/summary/{name}.txt'
    with open(summary_path, 'w') as f:
        f.write(f"{'Layer':<25} {'Type':<20} {'Output Shape':<25} {'Param #':<10}\n")
        f.write("="*80 + "\n")
        for layer in model.layers:
            try:
                output_shape = layer.output_shape
            except AttributeError:
                try:
                    output_shape = layer.get_output_shape_at(0)
                except Exception:
                    output_shape = "N/A"
            f.write(f"{layer.name:<25} {layer.__class__.__name__:<20} {str(output_shape):<25} {layer.count_params():<10}\n")
        f.write("="*80 + "\n")
        f.write(f"Total params: {model.count_params():,}\n")
    print(f"Saved summary: {summary_path}")