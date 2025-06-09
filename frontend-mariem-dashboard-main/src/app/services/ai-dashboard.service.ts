import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

@Injectable({
  providedIn: 'root'
})
export class AiDashboardService {
  private model: tf.Sequential | null = null;
  private trainingData: { inputs: number[], outputs: number[] }[] = [];

  constructor() { }

  // Initialise le modèle avec une architecture simple
  async initModel() {
    this.model = tf.sequential();
    
    // Couche d'entrée
    this.model.add(tf.layers.dense({
      units: 16,
      activation: 'relu',
      inputShape: [4] // 4 features: totalUsers, lockedAccounts, totalActions, lastWeekActions
    }));
    
    // Couche cachée
    this.model.add(tf.layers.dense({
      units: 8,
      activation: 'relu'
    }));
    
    // Couche de sortie
    this.model.add(tf.layers.dense({
      units: 3, // 3 catégories: normal, warning, danger
      activation: 'softmax'
    }));

    // Compilation du modèle
    this.model.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    console.log('Modèle initialisé');
  }

  // Prépare les données d'entraînement à partir des stats du dashboard
  prepareTrainingData(stats: any, activities: any[]) {
    // Exemple de données d'entraînement (dans un cas réel, vous auriez plus de données historiques)
    this.trainingData = [
      // Données normales
      { inputs: [10, 0, 50, 10], outputs: [1, 0, 0] }, // normal
      { inputs: [20, 1, 100, 20], outputs: [1, 0, 0] }, // normal
      
      // Données d'avertissement
      { inputs: [15, 3, 80, 30], outputs: [0, 1, 0] }, // warning
      { inputs: [30, 5, 150, 60], outputs: [0, 1, 0] }, // warning
      
      // Données dangereuses
      { inputs: [25, 8, 200, 100], outputs: [0, 0, 1] }, // danger
      { inputs: [40, 10, 300, 150], outputs: [0, 0, 1] }  // danger
    ];

    // Ajoutez les données actuelles pour l'entraînement
    const currentInput = [
      stats.totalUsers,
      stats.lockedAccounts,
      stats.totalActions,
      stats.lastWeekActions
    ];
    
    // Pour l'exemple, nous ajoutons les données actuelles comme "normales"
    this.trainingData.push({ 
      inputs: currentInput, 
      outputs: [1, 0, 0] 
    });
  }

  // Entraîne le modèle
  async trainModel() {
    if (!this.model) {
      await this.initModel();
    }

    // Prépare les tenseurs TensorFlow.js
    const inputs = this.trainingData.map(d => d.inputs);
    const outputs = this.trainingData.map(d => d.outputs);
    
    const inputTensor = tf.tensor2d(inputs);
    const outputTensor = tf.tensor2d(outputs);

    // Options d'entraînement
    const trainingOptions = {
      epochs: 100,
      validationSplit: 0.2,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Performance du Modèle' },
        ['loss', 'val_loss', 'acc', 'val_acc'],
        { callbacks: ['onEpochEnd'] }
      )
    };

    // Lance l'entraînement
    return this.model!.fit(inputTensor, outputTensor, trainingOptions);
  }

  // Prédit l'état du système
  async predictSystemStatus(stats: any): Promise<string> {
    if (!this.model) {
      await this.initModel();
      await this.trainModel();
    }

    const input = tf.tensor2d([[
      stats.totalUsers,
      stats.lockedAccounts,
      stats.totalActions,
      stats.lastWeekActions
    ]]);

    const prediction = this.model!.predict(input) as tf.Tensor;
    const result = await prediction.data();
    
    // Interprète les résultats
    const [normalProb, warningProb, dangerProb] = Array.from(result);
    
    if (dangerProb > 0.6) {
      return `DANGER (${(dangerProb * 100).toFixed(1)}% probabilité) - Le système montre des signes d'activité anormale`;
    } else if (warningProb > 0.5) {
      return `AVERTISSEMENT (${(warningProb * 100).toFixed(1)}% probabilité) - Surveillez les activités`;
    } else {
      return `NORMAL (${(normalProb * 100).toFixed(1)}% probabilité) - Tout semble fonctionner correctement`;
    }
  }

  // Analyse les tendances des activités
  analyzeActivityTrends(activities: any[]): string {
    // Simple analyse des tendances (pourrait être améliorée)
    const lastWeekCount = activities.filter(a => {
      const date = new Date(a.timestamp);
      const oneWeekAgo = new Date();
      oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
      return date > oneWeekAgo;
    }).length;

    const totalCount = activities.length;
    const ratio = lastWeekCount / (totalCount || 1);

    if (ratio > 0.5) {
      return `Activité élevée: ${lastWeekCount} actions cette semaine (${(ratio * 100).toFixed(0)}% du total)`;
    } else if (ratio > 0.3) {
      return `Activité normale: ${lastWeekCount} actions cette semaine`;
    } else {
      return `Activité faible: ${lastWeekCount} actions cette semaine`;
    }
  }

  // Détecte les anomalies de sécurité
  detectSecurityAnomalies(failedAttempts: any[]): string {
    if (failedAttempts.length === 0) {
      return "Aucune anomalie de sécurité détectée";
    }

    const recentAttempts = failedAttempts.slice(0, 5);
    const highRiskCount = recentAttempts.filter(a => {
      const attemptMatch = a.description.match(/Tentative (\d+)/);
      const attempts = attemptMatch ? parseInt(attemptMatch[1], 10) : 0;
      return attempts >= 3 || a.description.toLowerCase().includes('verrouillé');
    }).length;

    if (highRiskCount >= 3) {
      return `ANOMALIE DE SÉCURITÉ: ${highRiskCount} comptes à haut risque`;
    } else if (highRiskCount > 0) {
      return `Alerte sécurité: ${highRiskCount} tentative(s) suspecte(s)`;
    }

    return "Sécurité normale";
  }

  // Dans AiDashboardService

}