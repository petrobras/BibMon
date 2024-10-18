import bibmon
import bibmon.llm
import bibmon.llm.presets
import bibmon.llm.tools
import bibmon.three_w


def test_three_w_format_for_llm_prediction():
    df, ini, class_id = bibmon.load_3w()

    pre_processed = bibmon.PreProcess(
        f_pp=["remove_empty_variables", "ffill_nan", "remove_frozen_variables"]
    ).apply(df)

    formatted_data = bibmon.three_w.tools.format_for_llm_prediction(
        pre_processed, ini, class_id, 10
    )

    assert isinstance(formatted_data, dict)
    assert formatted_data.get("event_name") == "HYDRATE_IN_PRODUCTION_LINE"


def test_three_w_find_linked_column():
    df, ini, class_id = bibmon.load_3w()

    pre_processed = bibmon.PreProcess(
        f_pp=["remove_empty_variables", "ffill_nan", "remove_frozen_variables"]
    ).apply(df)

    formatted_data = bibmon.three_w.tools.format_for_llm_prediction(
        pre_processed, ini, class_id, 10
    )

    tool_schema, system_message, messages = (
        bibmon.llm.presets.three_w_find_linked_column(formatted_data)
    )

    assert tool_schema is None
    assert isinstance(system_message, bibmon.llm.types.LlmMessage)
    assert isinstance(messages, list)
