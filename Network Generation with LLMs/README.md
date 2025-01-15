# GUÍAS DE USO

## IMPORTANTE

Primeramente se ha de meter un token en el archivo `api-key.txt` para poder conectar con las APIs de los LLMs.

### Generación de APIs:
- [ChatGPT](https://platform.openai.com/)
- [Grok](https://console.x.ai/)
- [Llama](https://console.llamaapi.com/)
- [Gemini](https://aistudio.google.com/)
- [Claude](https://console.anthropic.com/)

---

## Descripción

### Para la generación de individuos:
Ejecute el siguiente comando:
```bash
python generate_personas.py 100 100_individuals
```

---

### Para la generación de redes:
Ejecute el siguiente comando:
```bash
python generate_networks.py sequential --model gpt-4o-mini --num_networks 11
```
Cambie el modelo (en este ejemplo, `gpt-4o-mini`) por el que usted desee.

⚠️ **WARNING** ⚠️  
Dependiendo del número de redes a generar (e.g., 11) y del modelo a utilizar (e.g., gpt-4o-mini), el proceso de generación puede implicar:
- **Lentitud**: El tiempo de ejecución puede variar desde unos pocos minutos hasta varias horas o incluso días.
- **Coste**: Los gastos pueden oscilar entre unos centimos de dólar hasta cientos de dólares (sí, dólares, estas empresas de IA facturan en dólares, no en euros), dependiendo del volumen de redes y del modelo seleccionado.  

Por favor, asegúrate de ajustar los parámetros con precaución y de considerar estos factores antes de ejecutar el proceso.

---

### Para generar los datos necesarios para analizar las redes generadas:
Ejecute el siguiente comando:
```bash
python analyze_networks.py --persona_fn 100_individuals.json --network_fn gpt-4o-mini --num_networks 11
```
Cambie el modelo (en este ejemplo, `gpt-4o-mini`) por el que usted desee.

---

### Para realizar y visualizar el análisis:
Revisa el siguiente notebook:
- `analyze_networks.ipynb`

## Referencias
- [LLMs Generate Structurally Realistic Social Networks but Overestimate Political Homophily](https://arxiv.org/abs/2408.16629)
- [Large Language Models for Social Networks: Applications, Challenges, and Solutions](https://arxiv.org/abs/2401.02575)
- [Network Formation and Dynamics Among Multi-LLMs](https://arxiv.org/html/2402.10659v3)
- [Marked Personas: Using Natural Language Prompts to Measure Stereotypes in Language Models](https://arxiv.org/abs/2305.18189)
- [Can Language Models Solve Graph Problems in Natural Language?](https://arxiv.org/abs/2305.10037)
- [Whose Opinions Do Language Models Reflect?](https://arxiv.org/abs/2303.17548)
- [A Long-Term Analysis of Polarization on Twitter](https://www.researchgate.net/publication/314361332_A_Long-Term_Analysis_of_Polarization_on_Twitter)
- [Large Language Model agents can coordinate beyond human scale](https://arxiv.org/abs/2409.02822)
- [A Large-scale Empirical Study on Large Language Models for Election Prediction](https://arxiv.org/abs/2412.15291)
