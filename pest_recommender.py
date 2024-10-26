from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import PIL.Image
import google.generativeai as genai
import base64


GOOGLE_API_KEY = "AIzaSyAXKYjauf98OPcjTkjsRHsqlBkv8kXI-PM" 
# Set API key manually
genai.configure(api_key="AIzaSyAXKYjauf98OPcjTkjsRHsqlBkv8kXI-PM")

class pest_recommender:
    def __init__(self):
        self.__model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, google_api_key=GOOGLE_API_KEY)
        self.__output_parser = StrOutputParser()
        self.__template = ("The {pest_name} pest are effect to the gherkin inderstry lot."
                           "consider above details, and recommend the following:"
                            "Introduce about {pest_name} pest"
                            "What are the best practices to prevent this pest inspection?"
                            "how to control spread of this pest?"
                           )
        
        self.__prompt_template = PromptTemplate(template=self.__template, input_variables=["pest_name"])

    # def load_and_encode_image(self, image_path: str) -> str:
    #     with open(image_path, "rb") as img_file:
    #         encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    #     return encoded_string

    def getRecommendations(self, pest_name: str) -> list[str]:
        chain = self.__prompt_template | self.__model | self.__output_parser
        recomndations = chain.invoke({"pest_name": pest_name})
        
        return recomndations
    # def getSeverity(self, disease_name: str, image_path: str) -> str:
    #     model = genai.GenerativeModel("gemini-1.5-flash")
    #     image_data = PIL.Image.open(image_path)
    #     response = model.generate_content([f"Analyze the input image and assess the severity of the {disease_name} present. Consider factors such as the extent of visible symptoms, color variations, texture changes, and any other relevant visual indicators. Provide a detailed severity score on a scale of 1 to 10, with 1 being mild and 10 being severe. Additionally, offer recommendations for possible treatment options or next steps based on the assessed severity.", image_data])
        
    #     return response.candidates[0].content.parts[0].text
    

# test = recommender().getRecommendations("Downt Mildew", 30.0, 70.0)
# serverity = recommender().getSeverity("Downt Mildew", "D:\\Campus\\SLIIT\\Y4\\Research Project\\amanda\\7.jpg")
# print(test)
# print("-"*50)
# print(serverity)