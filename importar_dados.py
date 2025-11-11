#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simplificado para análise de árvore de decisão.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
    X_encoded = X.copy()
    label_encoders = {}
    
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
            label_encoders[col] = le
    
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
