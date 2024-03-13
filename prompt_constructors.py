import json


# This file contains prompt constructors for various pieces of code. Used primarily to keep other code legible.
def construct_tool_use_system_prompt(tools):
    tool_use_system_prompt = (
            "In this environment you have access to a set of tools you can use to answer the user's question.\n"
            "\n"
            "You may call them like this:\n"
            "<function_calls>\n"
            "<invoke>\n"
            "<tool_name>$TOOL_NAME</tool_name>\n"
            "<parameters>\n"
            "<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>\n"
            "...\n"
            "</parameters>\n"
            "</invoke>\n"
            "</function_calls>\n"
            "\n"
            "Here are the tools available:\n"
            "<tools>\n"
            + '\n'.join(
        [construct_format_tool_for_claude_prompt(tool["function"]["name"], tool["function"]["description"],
                                                 tool["function"]["parameters"]["properties"]) for tool in
         tools]) +
            "\n</tools>"
    )

    return tool_use_system_prompt


def construct_successful_function_run_injection_prompt(invoke_results_results) -> str:
    constructed_prompt = (
            "<function_results>\n"
            + '\n'.join(
        f"<result>\n<tool_name>{res['tool_call_id']}</tool_name>\n<stdout>\n{res['content']}\n</stdout>\n</result>" for
        res in invoke_results_results) +
            "\n</function_results>"
    )

    return constructed_prompt


def construct_error_function_run_injection_prompt(invoke_results_error_message) -> str:
    constructed_prompt = (
        "<function_results>\n"
        "<system>\n"
        f"{invoke_results_error_message}"
        "\n</system>"
        "\n</function_results>"
    )

    return constructed_prompt


def construct_format_parameters_prompt(parameters) -> str:
    constructed_prompt = "\n".join(
        f"<parameter>\n<name>{key}</name>\n<type>{value['type']}</type>\n<description>{value['description']}</description>\n</parameter>"
        for key, value in parameters.items())

    return constructed_prompt


def construct_format_tool_for_claude_prompt(name, description, parameters) -> str:
    constructed_prompt = (
        "<tool_description>\n"
        f"<tool_name>{name}</tool_name>\n"
        "<description>\n"
        f"{description}\n"
        "</description>\n"
        "<parameters>\n"
        f"{construct_format_parameters_prompt(parameters)}\n"
        "</parameters>\n"
        "</tool_description>"
    )

    return constructed_prompt


def construct_tool_inputs_message(content, tool_inputs) -> str:
    def format_parameters(tool_arguments):
        return '\n'.join([f'<{key}>{value}</{key}>' for key, value in json.loads(tool_arguments).items()])

    single_call_messages = "\n\n".join([
        f"<invoke>\n<tool_name>{tool_input['tool_name']}</tool_name>\n<parameters>\n{format_parameters(tool_input['tool_arguments'])}\n</parameters>\n</invoke>"
        for tool_input in tool_inputs])
    message = (
        f"{content}"
        "\n\n<function_calls>\n"
        f"{single_call_messages}\n"
        "</function_calls>"
    )
    return message


def construct_tool_outputs_message(tool_outputs, tool_error) -> str:
    if tool_error is not None:
        message = construct_error_function_run_injection_prompt(tool_error)
        return f"\n\n{message}"
    elif tool_outputs is not None:
        message = construct_successful_function_run_injection_prompt(tool_outputs)
        return f"\n\n{message}"
    else:
        raise ValueError("At least one of tool_result or tool_error must not be None.")
