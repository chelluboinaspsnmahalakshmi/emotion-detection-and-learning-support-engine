from src.model import predict_bilstm
from src.bert_model import predict_bert
from src.preprocessing import get_mixed_emotions


def run_prediction(raw_text: str, bilstm_assets, bert_assets) -> dict:
    """
    Run prediction through both BiLSTM and BERT models.

    Args:
        raw_text: raw student input text
        bilstm_assets: tuple of (model, tokenizer, label_classes)
        bert_assets: tuple of (model, tokenizer, label_mapping)

    Returns:
        dict with keys:
            bilstm_result: {emotion, confidence, scores, cleaned_text}
            bert_result: {emotion, confidence, scores, cleaned_text}
            bilstm_mixed: list of (emotion, score) tuples above 15% threshold
            bert_mixed: list of (emotion, score) tuples above 15% threshold
    """
    bilstm_result = predict_bilstm(raw_text, *bilstm_assets)
    bert_result = predict_bert(raw_text, *bert_assets)

    bilstm_mixed = get_mixed_emotions(bilstm_result['scores'])
    bert_mixed = get_mixed_emotions(bert_result['scores'])

    return {
        'bilstm_result': bilstm_result,
        'bert_result': bert_result,
        'bilstm_mixed': bilstm_mixed,
        'bert_mixed': bert_mixed
    }
