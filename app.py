# gradio ui, reset, cleanups
import gradio as gr
from research_assistant import ResearchAssistant

async def setup():
    assistant = ResearchAssistant()
    await assistant.setup()
    return assistant

async def process_message(assistant, message, success_criteria, history):
    results = await assistant.run_superstep(message, success_criteria, history)
    return results +[assistant]

async def reset():
    new_assistant = ResearchAssistant()
    await new_assistant.setup()
    return "", "", None, None, None, new_assistant

async def free_resources(assistant):
    print("Cleaning up")
    try:
        if assistant:
            assistant.cleanup()
    except Exception as e:
        print(f"Exception during cleanup: {e}")


with gr.Blocks(theme=gr.themes.Default(primary_hue="emerald")) as demo:
    gr.Markdown("## Deep Research Assistant")
    assistant = gr.State(delete_callback=free_resources) #create new assistant object

    with gr.Row():
        with gr.Column():
            with gr.Row():
                chatbot = gr.Chatbot(label="Research Assistant", height=200, type="messages")
            with gr.Group():
                with gr.Row():
                    message = gr.Textbox(placeholder="Please enter your research query here.")
                with gr.Row():
                    success_criteria=gr.Textbox(placeholder="What are your success criteria?")

        with gr.Column():
            with gr.Row():
                topic=gr.Markdown()
            #with gr.Row():
                #status=gr.Markdown() # status is not doing what we want(only update at the end about final report generated)
            with gr.Row():
                feedback=gr.Markdown()
            with gr.Row():
                final_report=gr.Markdown(height=500)
    
    demo.load(setup, [], [assistant])
    with gr.Row():
        reset_button=gr.Button("Reset", variant="stop")
        go_button=gr.Button("Go!", variant="primary")

    go_button.click(process_message, [assistant, message, success_criteria, chatbot], [chatbot, topic, feedback, final_report, assistant])
    reset_button.click(reset, [], [message, success_criteria, chatbot, topic, final_report, assistant])

demo.launch()