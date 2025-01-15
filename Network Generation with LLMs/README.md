#IMPORTANTE: Primeramente se ha de meter un token  en el archivo api-key.tx. para poder conectar con las APIs de los LLMs 

Generacion de APIs:
- ChatGPT: https://platform.openai.com/
- Grok: https://console.x.ai/
- Llama: https://console.llamaapi.com/
- Gemini: https://aistudio.google.com/
- Claude: https://console.anthropic.com/


#Description
Para la generacion de individuos ejecute
python generate_personas.py 100 100_individuals


#Para la generacion de redes ejecute:
python generate_networks.py sequential --model gpt-4o-mini --num_networks 11
(cambie el modelo, en este ejemplo gpt-4o-mini, por el que usted desee

⚠️ WARNING ⚠️
Dependiendo del número de redes a generar (e.g., 50) y del modelo a utilizar (e.g., GPT-3.5-turbo), el proceso de generación puede implicar:
- Lentitud: El tiempo de ejecución puede variar desde unos pocos minutos hasta varias horas o incluso días.
- Coste: Los gastos pueden oscilar entre unos pocos centavos de dólar hasta cientos de dólares, dependiendo del volumen de redes y del modelo seleccionado.
Por favor, asegúrate de ajustar los parámetros con precaución y de considerar estos factores antes de ejecutar el proceso.


#Para general los datos necesarios para analizar las redes generadas:
python analyze_networks.py --persona_fn 100_individuals.json --network_fn gpt-4o-mini --num_networks 11
(cambie el modelo, en este ejemplo gpt-4o-mini, por el que usted desee)

#Para realizar y visualizar el analisis vea el siguiente Notebook:
-analyze_networks.pynb


