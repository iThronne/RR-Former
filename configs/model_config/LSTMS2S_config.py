from ..data_shape_config import DataShapeConfig


class LSTMS2SConfig:
    model_name = "LSTMS2S"
    decode_mode = None
    seq_len_e1 = DataShapeConfig.src_len
    output_len_e1 = DataShapeConfig.pred_len
    input_size_e1 = DataShapeConfig.src_size
    hidden_size_e1 = 128
    seq_len_e2 = DataShapeConfig.past_len
    input_size_e2 = DataShapeConfig.tgt_size
    hidden_size_e2 = 128
    output_len_d = DataShapeConfig.pred_len
    input_size_d = 256
    hidden_size_d = 128
    dropout_rate = 0.2

    model_info = f"{model_name}_[[{seq_len_e1}-{output_len_e1}-{input_size_e1}-{hidden_size_e1}]" \
                 f"[{seq_len_e2}-*-{input_size_e2}-{hidden_size_e2}]" \
                 f"[*-{output_len_d}-{input_size_d}-{hidden_size_d}]-{dropout_rate}]"
