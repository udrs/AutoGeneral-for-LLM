import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Type
import openai
from openai import OpenAI
import requests
from qwen_vl_utils import process_vision_info
from io import BytesIO
from PIL import Image

# 基础数据结构
@dataclass
class ThemeConfig:
    """主题配置数据结构"""
    main_theme: str
    sub_themes: List[str]
    interaction_type: str = "interaction"  # 新增交互类型字段

@dataclass
class DialogueTurn:
    """对话轮次数据结构"""
    round_number: int
    image_description: str
    user_prompt: str
    model_response: str
    verification_result: Optional[dict]

# 策略模式基类
class InteractionStrategy(ABC):
    @abstractmethod
    def generate_themes(self, client, num_themes=5) -> List[ThemeConfig]:
        pass
    
    @abstractmethod
    def build_dialogue_flow(self, theme: ThemeConfig, processor, model, client) -> List[DialogueTurn]:
        pass
    
    @abstractmethod
    def verify_response(self, current_response: str, history: List[str]) -> dict:
        pass

# 残留型策略
class ResidualStrategy(InteractionStrategy):
    def generate_themes(self, client, num_themes=5) -> List[ThemeConfig]:
        # 实现原有残留型主题生成逻辑
        pass
    
    def build_dialogue_flow(self, theme, processor, model, client) -> List[DialogueTurn]:
        # 实现原有对话流程
        pass
    
    def verify_response(self, current_response, history):
        # 实现原有验证逻辑
        pass

# 差异型交互策略
class DifferenceInteractionStrategy(InteractionStrategy):
    def __init__(self):
        # 策略特定模板配置
        self.image_generator = OpenAI(api_key="openaikey")
        # self.image_generator = OpenAI(api_key="openaikey")
        self.theme_templates = { 
            "Interaction": {
                "query": (
                    "生成需要主题，要求：\n"
                    "1. 某产品的使用说明书\n"
                    "2. 五个步骤\n"
                    "请严格按以下格式响应：\n"
                    "主题：[产品]\n"
                    "步骤1：[操作]\n "
                    "步骤2: [操作]\n "
                    "步骤3: [操作]\n"
                    "步骤4: [操作]\n"
                    "步骤5: [操作]\n"
                ),
                "examples": [
                    "主题：电子耳塞\n"
                    "步骤1: 插入耳塞\n"
                    "步骤2: 调整音量\n"
                    "步骤3: 佩戴舒适\n"
                ]
            }
        }
        self.path0 = None
        self.path1 = None
        self.path2 = None
        self.path3 = None
        self.path4 = None
        self.path_ask = None

        self.prompt0 = None
        self.prompt1 = None
        self.prompt2 = None
        self.prompt3 = None
        self.prompt4 = None
        

        self.A1 = None
        self.A2 = None

    def _parse_theme_response(self, response: str) -> dict:
        """解析主题响应文本为结构化数据
        

        """
        import re
        parsed = {}
        key_mapping = {
            "主题": "主题",
            "属性": "属性",
            "属性下具体特征": "属性下具体特征"
        }
        
        # 多分隔符处理
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # 支持中文冒号和英文冒号
            if '：' in line:
                key_part, value_part = line.split('：', 1)
            elif ':' in line:
                key_part, value_part = line.split(':', 1)
            else:
                continue
                
            key = key_part.strip()
            value = value_part.strip()
            
            # 映射合法字段
            if key in key_mapping:
                parsed_key = key_mapping[key]
                # 特殊处理特征字段
                if parsed_key == "属性下具体特征":
                    # 支持多种分隔符
                    parsed[parsed_key] = [x.strip() for x in re.split(r' vs |、|,', value)]
                else:
                    parsed[parsed_key] = value
                    
        return parsed



    def generate_themes(self, client, num_themes=10) -> List[ThemeConfig]:
        generated = []
        template = self.theme_templates["Interaction"]
        prompt = f"""模板：
    {template['query']}

    当前示例：
    {random.choice(template['examples'])}

    请严格按以下格式响应：
    主题：[产品]\n
    步骤1：[操作]\n
    步骤2: [操作]\n
    步骤3: [操作]\n
    步骤4: [操作]\n
    步骤5: [操作]"""
        
        # 用于记录已生成的主题
        past_themes = []
        
        try:
            while len(generated) < num_themes:
                # 将已生成的主题拼接到系统提示中
                system_prompt = f"你是专业的技术文档工程师，生成的主题不要与已有主题{past_themes}重复"
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    stream=False
                )
                
                raw_text = response.choices[0].message.content
                parsed = self._parse_theme_response(raw_text)
                required_fields = ['主题', '步骤1', '步骤2', '步骤3', '步骤4', '步骤5']
                if not all(key in parsed for key in required_fields):
                    print("响应缺少必要字段，跳过此轮")
                    continue
                
                theme = ThemeConfig(
                    main_theme=parsed['主题'],
                    sub_themes=[
                        parsed['步骤1'],
                        parsed['步骤2'],
                        parsed['步骤3'],
                        parsed['步骤4'],
                        parsed['步骤5'],
                    ],
                    interaction_type="logical"
                )
                
                # 如果生成的主题在历史中已存在，则跳过
                if theme.main_theme in past_themes:
                    print("生成的主题重复，重新生成")
                    continue
                
                generated.append(theme)
                past_themes.append(theme.main_theme)
            
        except Exception as e:
            print(f"ISSUE HAPPENED: {str(e)}")
        
        return generated



    # def generate_themes(self, client, num_themes=5) -> List[ThemeConfig]:
    #     generated = []
    #     template = self.theme_templates["Interaction"]
    #     # 构建动态提示词
    #     prompt = f"""模板：
    #     {template['query']}
        
    #     当前示例：
    #     {random.choice(template['examples'])}
        
    #     请严格按以下格式响应：
    #     主题：[产品]\n
    #     步骤1：[操作]\n
    #     步骤2: [操作]\n
    #     步骤3: [操作]\n
    #     步骤4: [操作]\n
    #     步骤5: [操作]"""
    #     try:
    #         for _ in range(num_themes):
    #             response = client.chat.completions.create(
    #                 model="deepseek-chat",
    #                 messages=[
    #                     {"role": "system", "content": "你是专业的技术文档工程师"},
    #                     {"role": "user", "content": prompt}
    #                 ],
    #                 temperature=0.7,
    #                 stream=False
    #             )

    #             raw_text = response.choices[0].message.content
    #             parsed = self._parse_theme_response(raw_text)

    #             required_fields = ['主题', '步骤1', '步骤2', '步骤3', '步骤4', '步骤5']
    #             if not all(key in parsed for key in required_fields):
    #                 raise ValueError("响应缺少必要字段")

    #             theme = ThemeConfig(
    #             main_theme=parsed['主题'],
    #             sub_themes=[
    #                 parsed['步骤1'],
    #                 parsed['步骤2'],
    #                 parsed['步骤3'],
    #                 parsed['步骤4'],
    #                 parsed['步骤5'],
    #             ],        
    #             interaction_type="logical"
    #             )
                
    #             generated.append(theme)
    #             return generated
    #             # generated.append(response)
    #     except Exception as e:
    #         print(f"ISSUE HAPPENED: {str(e)}")
    #     return generated
    
    def _parse_theme_response(self, response: str) -> dict:
        import re
        parsed = {}
        pattern = r"主题[:：]\s*(.*?)\n\n步骤1[:：]\s*(.*?)\n\n步骤2[:：]\s*(.*?)\n\n步骤3[:：]\s*(.*?)\n\n步骤4[:：]\s*(.*?)\n\n步骤5[:：]\s*(.*)"

        match = re.search(pattern, response)
        if match:
            parsed["主题"] = match.group(1).strip()
            parsed["步骤1"] = match.group(2).strip()
            parsed["步骤2"] = match.group(3).strip()
            parsed["步骤3"] = match.group(4).strip()
            parsed["步骤4"] = match.group(5).strip()
            parsed["步骤5"] = match.group(6).strip()
        return parsed

    def _generate_image(self, prompt: str, image_path: str, instruct: str,client) -> str:
        """Generates an image using DALL·E 3 and saves it locally."""
        try:
            instruct_EN = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "you are a translator, translate the chinese in English"},
                    {"role": "user", "content": instruct},
                ],
                temperature=0.7,  # 增加创造性
                top_p=0.9,
                stream=False
            )
            
            if instruct_EN.choices:
                Ins_EN = instruct_EN.choices[0].message.content.strip()
            
            # Request image generation from DALL·E 3
            response = self.image_generator.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",  # Adjust size if needed
                n=1,  # Number of images to generate
            )
            response.data[0].url
            # Get the image URL from the response
            image_url = response.data[0].url
            image_data = requests.get(image_url).content

            img = Image.open(BytesIO(image_data))

            img_resized = img.resize((256, 256))
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(img_resized)

            # Load a font (adjust font path if needed)
            try:
                font = ImageFont.truetype("arial.ttf", 20)  # Use system default font
            except IOError:
                font = ImageFont.load_default()  # Fallback to default if Arial not found

            # Define text position (bottom-left corner with padding)
            text_x = 10
            text_y = img_resized.height - 40  # Adjust to ensure visibility

            # Add text to image (black color)
            draw.text((text_x, text_y), Ins_EN, fill="black", font=font)

            # Save the final image
            img_resized.save(image_path)
            print(f"Image saved at: {image_path}")

            return image_path

        except Exception as e:
            print(f"Error generating image: {e}")
            return None

    def _generate_prompt(self,theme,client,time):
        # value = theme
        main_theme = theme.main_theme
        sub_theme = theme.sub_themes[time] 

        response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": f"Just refine this word, DON'T give any other responses:{main_theme} with attributes {sub_theme} "},
                    ],
                    stream=False
                )
        text_ = response.choices[0].message.content

        return text_


    def _generate_comparison_prompt(self,prompt1,prompt2,client,ite):
        if ite == 1:
            response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant,严格以'[属性]不同，[属性下Round1图片的具体特征]和[属性下Round2图片的具体特征]'"},
                            {"role": "user", "content": f"Tell us the difference between 'round1图片：{prompt1}' and 'round2图片：{prompt2}', don't give any other response, the answer should not exceed 8 words"},
                        ], 
                        stream=False
                    )
        else:
            response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant,严格以'[属性]不同，[属性下Round1图片的具体特征]和[属性下Round2图片的具体特征]'"},
                            {"role": "user", "content": f"Tell us the difference between 'round1图片：{prompt1}' and 'round2图片：{prompt2}', don't give any other response, the answer should not exceed 20 words"},
                        ], 
                        stream=False
                    )
        text_ = response.choices[0].message.content
        return text_

    def _generate_model_response(self, processor, model, messages) -> str:
        """Generate a response from the model."""
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


        try:
            image_inputs, video_inputs = process_vision_info(messages)
        except Exception as e:
            print(f"Warning: Error processing vision info: {e}. Skipping vision inputs.")
            image_inputs, video_inputs = None, None  # Fallback in case of error
        # image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs for the model
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            max_length=128,)
        
        inputs = inputs.to("cuda")

        # Generate model response
        generated_ids = model.generate(**inputs, max_new_tokens=128,temperature=0.3)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the output
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text



    def _create_message(self, round_number: int, path: str, question: str) -> dict:
        """Create a message dictionary for a given round."""
        return {
            "role": "user",
            "content": [
                {"type": "image", "image": path},
                {"type": "text", "text": question}
            ]
        }
    
    
    
    
    def adv(self, theme, client):
    # def enhance_attribute_difficulty(self, theme, client):
        """强化主题属性区分难度
        Args:
            theme: 包含主属性和子属性的主题对象
            client: 大模型API客户端
        Returns:
            theme: 增强难度后的主题对象
        """
        main_attribute = theme.main_theme

        # 优化后的系统提示词
        system_prompt = """ 生成需要主题，要求：
        1. 特征冷门化：使用专业术语、小众领域知识或罕见表达方式
        2. 特征模糊化：在相似特征中制造微妙差异
        3. 跨领域融合：结合非常规的跨界元素
        4. 格式严格遵循：
        [属性]:[Round1特征]||[Round2特征]
        示例：
        [北极天空天文现象]:[磁暴引发的极光畸变]||[太阳风异常导致的极光红化]"""

        # 优化后的用户提示词
        user_prompt = f"""请处理主属性「{main_attribute}」：
        - 生成相似但存在细微差异的特征
        - 使用非典型场景/非常规角度/专业术语
        - 确保特征描述在语义空间上接近但本质不同
        输出格式：{main_attribute}:特征1||特征2"""

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,  # 增加创造性
                top_p=0.9,
                stream=False
            )
            
            if response.choices:
                generated_text = response.choices[0].message.content.strip()
                # 添加解析逻辑示例（根据实际格式调整）
                if '||' in generated_text:
                    theme.sub_themes[0], theme.sub_themes[1] = generated_text.split('||')
            return theme
            
        except Exception as e:
            print(f"API调用异常: {str(e)}")
            return theme  # 返回原始theme作为降级处理

    def parseQandA(self,Hard_AandQ):
        # 假设 Hard_AandQ 的格式是 "Q:[问题] | A:[答案]"
        if "\n" in Hard_AandQ:
            q_part, a_part = Hard_AandQ.split("\n", 1)
            
            # 提取问题部分
            if "Q:" in q_part:
                question = q_part.split("Q:", 1)[1].strip()
            else:
                question = ""
            
            # 提取答案部分
            if "A:" in a_part:
                answer = a_part.split("A:", 1)[1].strip()
            else:
                answer = ""
            
            return question, answer
        else:
            # 如果没有 "|"，则返回空字符串
            return "", ""

    def build_dialogue_flow(self, theme, processor, model, client,ite) -> List[DialogueTurn]:
        messages = []

        # adv setting
        if ite>=2:
            messages = []
          

            # ask question
            messages.append(self._create_message(0, self.path0, f"这是{theme.main_theme}使用说明书的一部分"))
            round_0_response = self._generate_model_response(processor, model, messages)
            messages.append({"role": "Assistant", "content": [{"type": "text", "text": round_0_response}]})

        
            messages.append(self._create_message(1, self.path2, f"这是{theme.main_theme}使用说明书的一部分"))
            round_1_response = self._generate_model_response(processor, model, messages)
            messages.append({"role": "Assistant", "content": [{"type": "text", "text": round_1_response}]})

        
            messages.append(self._create_message(2, self.path1, f"这是{theme.main_theme}使用说明书的一部分"))
            round_2_response = self._generate_model_response(processor, model, messages)
            messages.append({"role": "Assistant", "content": [{"type": "text", "text": round_2_response}]})

            messages.append(self._create_message(3, self.path4, f"这是{theme.main_theme}使用说明书的一部分"))
            round_3_response = self._generate_model_response(processor, model, messages)
            messages.append({"role": "Assistant", "content": [{"type": "text", "text": round_3_response}]})

            messages.append(self._create_message(4, self.path3, f"这是{theme.main_theme}使用说明书的一部分"))
            round_4_response = self._generate_model_response(processor, model, messages)
            messages.append({"role": "Assistant", "content": [{"type": "text", "text": round_4_response}]})

            instrctions = f"产品主题：{theme.main_theme} 步骤1：{theme.sub_themes[0]} 步骤2：{theme.sub_themes[1]} 步骤3：{theme.sub_themes[2]} 步骤4：{theme.sub_themes[3]} 步骤5：{theme.sub_themes[4]}"

            AandQ = client.chat.completions.create(
                model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": f"这是说明书：{instrctions}。问一个需要结合多个步骤理解的问题.严格遵循'Q:[问题] \n A:[答案]'回答"},
                        {"role": "user", "content": f"根据说明书的内容，问一个需要通过多步骤结合理解的非常难的问题。并且给一个分明确几个步骤的答案"},
                    ],
                    stream=False
                )
            Hard_QandA = AandQ.choices[0].message.content
            Qes, Ans = self.parseQandA(Hard_QandA)
            # prompt_ask = """我已经完成了第一步，要使用该产品，讲一下接下来要怎么做，
            # 请严格按以下格式响应：
            # 主题：[产品]
            # 步骤[times]：[操作]
            # 步骤[times]：[操作]"""
            messages.append(self._create_message(5, self.path_ask, Qes))
            round_5_response = self._generate_model_response(processor, model, messages)
            self.A1 = round_5_response
            messages.append({"role": "Assistant", "content": [{"type": "text", "text": round_5_response}]})
            print(self.A1)
            # self.A2 = f'主题: {theme.main_theme}\n 步骤1: {theme.sub_themes[1]}\n 步骤3: {theme.sub_themes[2]}'            
            print(messages)
            self.A2 = Ans
            return messages,self.A1,self.A2




        from datetime import datetime
        import os
        from PIL import Image, ImageDraw, ImageFont

        path = os.getcwd()
        # prepare for image 
        self.prompt0 = f"{theme.main_theme}使用说明书第一步:{theme.sub_themes[0]}。画出'{theme.sub_themes[0]}'这一步,整体用淡色，禁用黑色"
        self.prompt1 = f"{theme.main_theme}使用说明书第二步:{theme.sub_themes[1]}。画出'{theme.sub_themes[1]}'这一步,整体用淡色，禁用黑色"
        self.prompt2 = f"{theme.main_theme}使用说明书第三步:{theme.sub_themes[2]}。画出'{theme.sub_themes[2]}'这一步,整体用淡色，禁用黑色"
        self.prompt3 = f"{theme.main_theme}使用说明书第四步:{theme.sub_themes[3]}。画出'{theme.sub_themes[3]}'这一步,整体用淡色，禁用黑色"
        self.prompt4 = f"{theme.main_theme}使用说明书第五步:{theme.sub_themes[4]}。画出'{theme.sub_themes[4]}'这一步,整体用淡色，禁用黑色"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path0 = self._generate_image(self.prompt0,path+f"/image_{timestamp}.png",f"1.{theme.sub_themes[0]}",client)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path1 = self._generate_image(self.prompt1,path+f"/image_{timestamp}.png",f"2.{theme.sub_themes[1]}",client)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path2 = self._generate_image(self.prompt2,path+f"/image_{timestamp}.png",f"3.{theme.sub_themes[2]}",client)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path3 = self._generate_image(self.prompt3,path+f"/image_{timestamp}.png",f"4.{theme.sub_themes[3]}",client)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path4 = self._generate_image(self.prompt4,path+f"/image_{timestamp}.png",f"5.{theme.sub_themes[4]}",client)

        image = Image.new('RGB', (256, 256), (255, 255, 255))    
        self.path_ask = os.path.join(path,"white_image.png")
        image.save(self.path_ask)
        


        # ask question
        messages.append(self._create_message(0, self.path0, f"这是{theme.main_theme}使用说明书的第一步"))
        round_0_response = self._generate_model_response(processor, model, messages)
        messages.append({"role": "Assistant", "content": [{"type": "text", "text": round_0_response}]})

     
        messages.append(self._create_message(1, self.path1, f"这是{theme.main_theme}使用说明书的第二步"))
        round_1_response = self._generate_model_response(processor, model, messages)
        messages.append({"role": "Assistant", "content": [{"type": "text", "text": round_1_response}]})

    
        messages.append(self._create_message(2, self.path2, f"这是{theme.main_theme}使用说明书的第三步"))
        round_2_response = self._generate_model_response(processor, model, messages)
        messages.append({"role": "Assistant", "content": [{"type": "text", "text": round_2_response}]})


        messages.append(self._create_message(3, self.path3, f"这是{theme.main_theme}使用说明书的第四步"))
        round_3_response = self._generate_model_response(processor, model, messages)
        messages.append({"role": "Assistant", "content": [{"type": "text", "text": round_3_response}]})


        messages.append(self._create_message(4, self.path4, f"这是{theme.main_theme}使用说明书的第五步"))
        round_4_response = self._generate_model_response(processor, model, messages)
        messages.append({"role": "Assistant", "content": [{"type": "text", "text": round_4_response}]})



        # forgetting hallucination
        prompt_ask = f"""介绍一下{theme.main_theme}的使用说明，
        请严格按以下格式响应：
        主题：[产品]
        步骤[steps]：[操作]
        步骤[steps]: [操作]
        步骤[steps]: [操作]"""
        messages.append(self._create_message(5, self.path_ask, prompt_ask))
        round_5_response = self._generate_model_response(processor, model, messages)
        self.A1 = round_5_response
        messages.append({"role": "Assistant", "content": [{"type": "text", "text": round_5_response}]})

        print(messages)
        self.A2 = f'主题: {theme.main_theme}\n 步骤1: {theme.sub_themes[0]}\n 步骤2: {theme.sub_themes[1]}\n 步骤3: {theme.sub_themes[2]}\n 步骤4: {theme.sub_themes[3]}\n 步骤5: {theme.sub_themes[4]}'

        return messages,self.A1,self.A2
    
    def verify_response(self, current_response, history):
        # 验证差异比较的准确性
        return {"difference_accuracy": 0.8}


class HallucinationDatasetGenerator:
    def __init__(self, strategy: Type[InteractionStrategy] = ResidualStrategy):
        self.strategy = strategy()
        self.theme_pool = []
        self.generated_data = []
        self.image_generator = OpenAI(api_key="your-api-key")
        
        # 初始化策略相关配置
        self._init_strategy_config()

    def _init_strategy_config(self):
        """初始化策略特定配置"""
        self.strategy_config = {
            "difference": {
                "theme_template": "生成对比主题模板",
                "verification_params": ["差异准确性"]
            },
            "logical": {
                "theme_template": "生成步骤说明模板",
                "verification_params": ["步骤完整性"]
            }
        }

    def set_strategy(self, strategy: Type[InteractionStrategy]):
        """动态切换策略"""
        self.strategy = strategy()
        self._init_strategy_config()

    def generate_themes(self, client, num_themes=5):
        """使用当前策略生成主题"""
        self.theme_pool = self.strategy.generate_themes(client, num_themes)
        return self.theme_pool

    def execute_test_case(self, theme: ThemeConfig, processor, model, client,ite):
        """使用当前策略执行测试用例"""
        ite = ite + 1
        history,A1,GT =  self.strategy.build_dialogue_flow(theme, processor, model, client, ite)
        print("PRE:",A1)
        print("GT:",GT)

        # Generate questions
        # 7B
        # prompt = f"""你是一个判断系统：回答正确程度80%以上则返回[YES],否则返回[NO]，不要回答额外信息"""

        # 72B
        prompt = f"""你是一个判断系统：回答正确则返回[YES],否则返回[NO]，不要回答额外信息"""
        answer = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"GT:{GT}是正确答案，请问我们的回答:{A1}是正确的吗"},
            ],
            stream=False
        )
        ansres = answer.choices[0].message.content
        print(ansres)
        if "Y" in ansres:
            history = self.execute_test_case(theme,processor,model,client,ite)
        else:
            return history
        return history, GT


    def _generate_image(self, prompt: str) -> str:
        """通用图片生成方法"""
        try:
            response = self.image_generator.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024"
            )
            return response.data[0].url
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

    def _call_vision_model(self, prompt: str, image_url: str) -> str:
        """通用模型调用方法"""
        # 实现实际的模型调用逻辑
        return "模拟响应"

    def generate_dataset(self, num_themes=5):
        """主生成流程"""
        client = OpenAI(api_key="dskey", base_url="https://api.deepseek.com")
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from qwen_vl_utils import process_vision_info
        import torch


        # TEST FOR 72B
        max_memory = {
            0: "20GB",  
            1: "20GB",
            2: "20GB",
            3: "20GB",
            "cpu": "100GB"
        }
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-72B-Instruct", torch_dtype=torch.bfloat16,
                        device_map="auto", attn_implementation="flash_attention_2",max_memory=max_memory
            )



        # TEST FOR 7B
        # model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16,
        #                 device_map="auto",
        #     )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct") 
        # 使用当前策略生成主题
        theme_pool = self.generate_themes(client, num_themes)
        import os
        # 如果文件存在，先读取已有数据
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        else:
            dataset = []
        
        # 执行所有测试用例
        for theme in theme_pool:
            ite = 0
            dialogue,GT = self.execute_test_case(theme,processor,model,client,ite)
            # 提取用于保存的格式
            dataset_entry = {
                "history": dialogue,
                "ground_truth": GT
            }
            dataset.append(dataset_entry)

        # 保存为 JSON 文件
        output_file = "airfryer_multimodal_dataset.json"
        with open(output_file, "a", encoding="utf-8") as f:
            json.dump(dataset_entry, f, ensure_ascii=False, indent=2)

        print(f"Dataset saved to {output_file}")
        
        self._save_dataset()
        return self.generated_data

    def _save_dataset(self, filename="hallucination_dataset.json"):
        """保存数据集"""
        with open(filename, 'w') as f:
            json.dump(self.generated_data, f, indent=2)

# 使用示例
if __name__ == "__main__":
    # 创建差异型交互数据集
    diff_generator = HallucinationDatasetGenerator(DifferenceInteractionStrategy)
    diff_generator.generate_dataset(2)
