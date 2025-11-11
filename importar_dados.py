#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def aplicar_replace(dados):
    """Converte valores categóricos para números usando mapeamento específico."""
    dados_encoded = dados.copy()
    
    # Mapeamento de valores categóricos para números
    mapeamentos = {
        # Sim/Não
        'Sim': 1,
        'Não': 2,
        
        # Finalidade
        'Industrial': 1,
        'Comercial': 2,
        'Doméstico': 3,
        
        # Capacidade (Kg por minuto)
        'Até 1Kg': 1,
        'Até 6.5Kg': 2,
        'Até 9Kg': 3,
        'Acima de 10Kg': 4,
        
        # Voltagem
        '127V': 1,
        '220V': 2,
        'Trifásico': 3,
        
        # Tipo de material
        'Embutidos': 1,
        'Carne, Frango': 2,
        'Diversos (Castanhas, Frutas, Graõs, Etc)': 3,
        
        # Orçamento
        'Até R$ 2,500,00': 1,
        'Até R$ 15,000,00': 2,
        'Acima de R$ 15,000,00': 3,
        
        # Potência
        'Até 0,25kW': 1,
        'Até 2,2kW': 2,
        'Até 5,5kW': 3,
        'Até 7,5kW': 4,
    }
    
    # Aplica o replace em todas as colunas
    for col in dados_encoded.columns:
        dados_encoded[col] = dados_encoded[col].replace(mapeamentos)
    
    return dados_encoded

def main():
    # 1. Importar dados
    df = pd.read_csv('base-dados-atualizada.csv', encoding='utf-8')
    df = df.drop(columns=['Perguntas'], errors='ignore')
    
    # 2. Dividir X e Y
    X = df.iloc[:, :-1]  # Features (todas exceto a última)
    y = df.iloc[:, -1]   # Target (última coluna)
    
    print(f"Total de registros: {len(df)}")
    print(f"Features: {len(X.columns)} colunas")
    print(f"Classes: {y.nunique()} categorias\n")
    
    # 3. Converter dados categóricos para numéricos
    # Primeiro aplica replace nos valores conhecidos
    X_encoded = aplicar_replace(X)
    
    # Depois aplica LabelEncoder nos valores que ainda são strings
    label_encoders = {}
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
            label_encoders[col] = le
    
    # Converte Y (produtos) com LabelEncoder
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    
    # 4. Treinar árvore de decisão
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_encoded, y_encoded)
    
    # Calcular acurácia no conjunto completo
    accuracy = clf.score(X_encoded, y_encoded)
    print(f"Acurácia: {accuracy*100:.2f}%\n")
    
    # 5. Gerar matriz de confusão
    y_pred = clf.predict(X_encoded)
    cm = confusion_matrix(y_encoded, y_pred)
    
    # Visualizar matriz de confusão
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_y.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Matriz de Confusão')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png', dpi=300, bbox_inches='tight')
    print("Matriz de confusão salva: matriz_confusao.png")
    plt.close()
    
    # 6. Visualizar árvore de decisão
    plt.figure(figsize=(20, 10))
    plot_tree(clf, 
              feature_names=X_encoded.columns, 
              class_names=le_y.classes_, 
              filled=True, 
              rounded=True,
              fontsize=10)
    plt.savefig('arvore_decisao.png', dpi=300, bbox_inches='tight')
    print("Árvore de decisão salva: arvore_decisao.png")
    plt.close()
    
    print("\nProcesso concluído!")

if __name__ == "__main__":
    main()
