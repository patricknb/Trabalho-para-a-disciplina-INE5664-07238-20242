Aprendizado de Máquina: Projeto Final - Rede Neural Artificial (RNA) 2024
📖 Descrição do Projeto

Este repositório contém o projeto final da disciplina de Redes Neurais, cujo objetivo é implementar, em Python, uma Rede Neural Artificial (RNA) do zero, utilizando apenas recursos de baixo nível. A implementação foi desenvolvida com base nos conceitos teóricos e matemáticos abordados em aula.
Funcionalidades Implementadas

A RNA implementada inclui os requisitos minimos indicados abaixo:

    Estrutura da rede: suporte para múltiplas camadas e pesos ajustáveis.
    Funções de ativação: três funções de ativação diferentes.
    Funções de perda: duas funções de perda distintas.
    Algoritmo de retropropagação (backpropagation): cálculo eficiente do gradiente.
    Otimização por gradiente descendente: ajuste iterativo dos pesos.

Modelos Desenvolvidos

Foram treinados três modelos de predição, utilizando conjuntos de dados públicos, para diferentes tarefas:

    Classificação Binária
    Dataset: Water Quality and Potability
    
    Classificação Multiclasse
    Dataset: Video Games Rating by ESRB
    
    Regressão
    Dataset: Home Value Insights

Os modelos foram avaliados com métricas apropriadas, garantindo erros no conjunto de teste inferiores a 50%.
💡 Integrantes

    Fernando Carlos Pereira (16105548)
    Patrick do Nascimento Bueno (20100864)

📚 Requisitos de Uso

Para executar o projeto, é necessário ter instalado:

    Python 3.8+
    Bibliotecas: NumPy, Pandas, scikit-learn, Matplotlib


💻 Como Usar
Rodando os Scripts

    Edite os parâmetros no arquivo correspondente à tarefa desejada:
        classificacao_binaria.py
        classificacao_multiclasse.py
        regressao.py
    Execute o script:

    python3 classificacao_binaria.py

Usando os Notebooks

    Faça o download dos notebooks da pasta e abra-os no Google Colab(ou plataforma similar).
    Carregue os arquivos de datasets na mesma pasta.
    Execute as células do notebook para treinar e avaliar os modelos.

🗂 Estrutura do Repositório

        /
        ├── __pycache__          
        ├── data/                # Conjuntos de datasets utilizados
        ├── old/                # Implementações anteriores e não funcionais
        ├── classificacao_binaria.py                 # Implementação do treino e teste de modelo de classificação binária        
        ├── classificacao_multiclasse.py                # Implementação do treino e teste de modelo de classificação multiclasse
        ├── funcoes_ativacao_perda.py                # Funções de ativação utilizadas pelo modelo de classificação binária
        ├── notebook_classificacao_binaria.ipynb                # Notebook com treino e teste de modelo de classificação binária   
        ├── notebook_classificacao_binaria_com_saidas.ipynb                # Notebook com treino e teste de modelo de classificação binária já executado
        ├── notebook_classificacao_multiclasse.ipynb                # Notebook com treino e teste de modelo de classificação multiclasse
        ├── notebook_classificacao_multiclasse_com_saida.ipynb                # Notebook com treino e teste de modelo de classificação multiclasse já executado
        ├── notebook_regressao.ipynb                # Notebook com treino e teste de modelo de regressão
        ├── notebook_regressao_com_saida.ipynb                # Notebook com treino e teste de modelo de regressão já executado
        ├── regressao.py                # Implementação do treino e teste do modelo de regressão 
        └── README.md                # Documentação


📝 Observações Importantes

    Certifique-se de utilizar os datasets disponibilizados na pasta /data ou mencionados acima.
    Para dúvidas ou problemas, entre em contato com os integrantes do projeto.
