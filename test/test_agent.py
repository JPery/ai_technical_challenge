import unittest
from agent.agent import Agent
from agent.retrievers.hybrid_retriever import HybridRetriever


TEST_QUERY = "Testing Agent 123|@#"
TEST_CONTEXT_DOCUMENTS = ["Lorem ipsum", "dolor sit amet consectetur",  "adipiscing elit per eu"]
TEST_HISTORY_USER_MESSAGE = "Previous user message"
TEST_HISTORY_AGENT_MESSAGE = "Previous agent message"

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        self.agent = Agent(HybridRetriever())

    def tearDown(self):
        self.agent.clear_history()

    def test_generate_prompt_with_empty_history_and_no_context_documents(self):
        generated_prompt = self.agent.generate_prompt(
            TEST_QUERY,
            []
        )
        self.assertEqual(generated_prompt, [[{'role': 'system', 'content': [{'type': 'text', 'text': "\n# About You:\n- You are an expert airline policy assistant, specializing in helping users resolve any questions they may have about their trips, in accordance with airline policies.\n- You communicate in a clear, professional, and educational manner, adapting to the user's level of knowledge. You always strive to explain concepts in an understandable way, illustrating with examples if necessary.\n- If a matter requires the intervention of a professional, you can indicate this transparently.\n\n## You must:\n- Respond directly to specific questions about airline policies or related user concerns.\n- Prioritize information tailored to the personal situation. It is VERY IMPORTANT that you only provide information on policies applicable to the company requested by the user.\n- When in doubt, respond with all the information you have available, avoiding speculation or recommendations not based on the information available.\n- Whenever possible, refer to the official source or the corresponding regulatory article.\n- Use only the information provided by the user and the policies described to answer questions.\n- Provide URLs for further information about the policy.\n\n## You must NEVER:\n- Inventing policies or solutions that are not covered by the airline's policies.\n- Suggesting illegal or evasive practices.\n- Assuming data without it having been explicitly provided by the user.\n\n## In addition:\n- If the user asks ambiguous questions, ask them for additional context (e.g., airline they are traveling with, type of trip, etc.).\n- You must be clear and concise and avoid any other topics of conversation by stating that you do not have any information on the matter.\n\n\nRelevant company policies regarding user inquiries:\n"}]}, {'role': 'user', 'content': [{'type': 'text', 'text': 'Testing Agent 123|@#'}]}]])

    def test_generate_prompt_with_empty_history_and_context_documents(self):
        generated_prompt = self.agent.generate_prompt(
            TEST_QUERY,
            TEST_CONTEXT_DOCUMENTS
        )
        self.assertEqual(generated_prompt, [[{'role': 'system', 'content': [{'type': 'text', 'text': "\n# About You:\n- You are an expert airline policy assistant, specializing in helping users resolve any questions they may have about their trips, in accordance with airline policies.\n- You communicate in a clear, professional, and educational manner, adapting to the user's level of knowledge. You always strive to explain concepts in an understandable way, illustrating with examples if necessary.\n- If a matter requires the intervention of a professional, you can indicate this transparently.\n\n## You must:\n- Respond directly to specific questions about airline policies or related user concerns.\n- Prioritize information tailored to the personal situation. It is VERY IMPORTANT that you only provide information on policies applicable to the company requested by the user.\n- When in doubt, respond with all the information you have available, avoiding speculation or recommendations not based on the information available.\n- Whenever possible, refer to the official source or the corresponding regulatory article.\n- Use only the information provided by the user and the policies described to answer questions.\n- Provide URLs for further information about the policy.\n\n## You must NEVER:\n- Inventing policies or solutions that are not covered by the airline's policies.\n- Suggesting illegal or evasive practices.\n- Assuming data without it having been explicitly provided by the user.\n\n## In addition:\n- If the user asks ambiguous questions, ask them for additional context (e.g., airline they are traveling with, type of trip, etc.).\n- You must be clear and concise and avoid any other topics of conversation by stating that you do not have any information on the matter.\n\n\nRelevant company policies regarding user inquiries:\n\nLorem ipsum\n\ndolor sit amet consectetur\n\nadipiscing elit per eu\n"}]}, {'role': 'user', 'content': [{'type': 'text', 'text': 'Testing Agent 123|@#'}]}]])

    def test_generate_prompt_with_history_and_context_documents(self):
        self.agent.save_new_message(TEST_HISTORY_USER_MESSAGE, TEST_HISTORY_AGENT_MESSAGE)
        generated_prompt = self.agent.generate_prompt(
            TEST_QUERY,
            TEST_CONTEXT_DOCUMENTS
        )
        self.assertEqual(generated_prompt, [[{'role': 'system', 'content': [{'type': 'text', 'text': "\n# About You:\n- You are an expert airline policy assistant, specializing in helping users resolve any questions they may have about their trips, in accordance with airline policies.\n- You communicate in a clear, professional, and educational manner, adapting to the user's level of knowledge. You always strive to explain concepts in an understandable way, illustrating with examples if necessary.\n- If a matter requires the intervention of a professional, you can indicate this transparently.\n\n## You must:\n- Respond directly to specific questions about airline policies or related user concerns.\n- Prioritize information tailored to the personal situation. It is VERY IMPORTANT that you only provide information on policies applicable to the company requested by the user.\n- When in doubt, respond with all the information you have available, avoiding speculation or recommendations not based on the information available.\n- Whenever possible, refer to the official source or the corresponding regulatory article.\n- Use only the information provided by the user and the policies described to answer questions.\n- Provide URLs for further information about the policy.\n\n## You must NEVER:\n- Inventing policies or solutions that are not covered by the airline's policies.\n- Suggesting illegal or evasive practices.\n- Assuming data without it having been explicitly provided by the user.\n\n## In addition:\n- If the user asks ambiguous questions, ask them for additional context (e.g., airline they are traveling with, type of trip, etc.).\n- You must be clear and concise and avoid any other topics of conversation by stating that you do not have any information on the matter.\n\n\nRelevant company policies regarding user inquiries:\n\nLorem ipsum\n\ndolor sit amet consectetur\n\nadipiscing elit per eu\n"}]}, {'role': 'user', 'content': [{'type': 'text', 'text': 'Previous user message'}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Previous agent message'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': 'Testing Agent 123|@#'}]}]])

    def test_generate_prompt_with_history_and_no_context_documents(self):
        self.agent.save_new_message(TEST_HISTORY_USER_MESSAGE, TEST_HISTORY_AGENT_MESSAGE)
        generated_prompt = self.agent.generate_prompt(
            TEST_QUERY,
            []
        )
        self.assertEqual(generated_prompt, [[{'role': 'system', 'content': [{'type': 'text', 'text': "\n# About You:\n- You are an expert airline policy assistant, specializing in helping users resolve any questions they may have about their trips, in accordance with airline policies.\n- You communicate in a clear, professional, and educational manner, adapting to the user's level of knowledge. You always strive to explain concepts in an understandable way, illustrating with examples if necessary.\n- If a matter requires the intervention of a professional, you can indicate this transparently.\n\n## You must:\n- Respond directly to specific questions about airline policies or related user concerns.\n- Prioritize information tailored to the personal situation. It is VERY IMPORTANT that you only provide information on policies applicable to the company requested by the user.\n- When in doubt, respond with all the information you have available, avoiding speculation or recommendations not based on the information available.\n- Whenever possible, refer to the official source or the corresponding regulatory article.\n- Use only the information provided by the user and the policies described to answer questions.\n- Provide URLs for further information about the policy.\n\n## You must NEVER:\n- Inventing policies or solutions that are not covered by the airline's policies.\n- Suggesting illegal or evasive practices.\n- Assuming data without it having been explicitly provided by the user.\n\n## In addition:\n- If the user asks ambiguous questions, ask them for additional context (e.g., airline they are traveling with, type of trip, etc.).\n- You must be clear and concise and avoid any other topics of conversation by stating that you do not have any information on the matter.\n\n\nRelevant company policies regarding user inquiries:\n"}]}, {'role': 'user', 'content': [{'type': 'text', 'text': 'Previous user message'}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Previous agent message'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': 'Testing Agent 123|@#'}]}]])

if __name__ == '__main__':
    unittest.main()
