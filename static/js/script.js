document.addEventListener('DOMContentLoaded', () => {
    // Éléments DOM
    const elements = {
        form: document.getElementById('configForm'),
        runBtn: document.getElementById('runBtn'),
        stopBtn: document.getElementById('stopBtn'),
        status: document.getElementById('status'),
        statusText: document.querySelector('.status-text'),
        statusIcon: document.querySelector('.status-icon'),
        progressFill: document.getElementById('progressFill'),
        progressText: document.getElementById('progressText'),
        resultsTableBody: document.querySelector('#resultsTable tbody'),
        resultsCount: document.getElementById('resultsCount'),
        downloadBtn: document.getElementById('downloadBtn'),
        chartOverlay: document.getElementById('chartOverlay'),
        phases: document.querySelectorAll('.phase'),
        phaseLines: document.querySelectorAll('.phase-line')
    };

    // Variables d'état
    let state = {
        chart: null,
        eventSource: null,
        isOptimizing: false,
        currentGen: -1,
        totalGen: 0,
        paretoData: [],
        allPoints: [] // Stocker tous les points pour ajuster les axes
    };

    // Initialisation
    initChart();
    setupEventListeners();

    function setupEventListeners() {
        elements.stopBtn.addEventListener('click', handleStop);
        elements.form.addEventListener('submit', handleSubmit);
        elements.downloadBtn.addEventListener('click', handleDownload);
    }

    async function handleStop() {
        if (!state.isOptimizing) return;
        
        try {
            await fetch('/api/stop', { method: 'POST' });
            updateStatus('⏹️', 'Optimisation arrêtée', 'warning');
            
            if (state.eventSource) state.eventSource.close();
            resetOptimization();
        } catch (error) {
            console.error('Erreur:', error);
            updateStatus('❌', 'Erreur lors de l\'arrêt', 'error');
        }
    }

    async function handleSubmit(e) {
        e.preventDefault();
        if (state.isOptimizing) {
            updateStatus('⚠️', 'Une optimisation est déjà en cours', 'warning');
            return;
        }

        // Réinitialiser
        resetState();
        state.isOptimizing = true;
        
        // Mettre à jour l'interface
        updateUIForStart();
        
        // Récupérer les données
        const formData = new FormData(elements.form);
        const config = {
            dataset: formData.get('dataset'),
            ngen: parseInt(formData.get('ngen')),
            pop_size: parseInt(formData.get('pop_size'))
        };

        try {
            const response = await fetch('/api/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            const result = await response.json();
            if (result.status === 'started') {
                updateStatus('🚀', 'Optimisation démarrée...', 'info');
                connectSSE();
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('Erreur:', error);
            updateStatus('❌', `Erreur: ${error.message}`, 'error');
            resetOptimization();
        }
    }

    function resetState() {
        state = {
            chart: state.chart,
            eventSource: null,
            isOptimizing: false,
            currentGen: -1,
            totalGen: 0,
            paretoData: [],
            allPoints: []
        };
        
        if (state.chart) {
            state.chart.data.datasets = [];
            state.chart.options.scales.x.min = 0;
            state.chart.options.scales.x.max = 10;
            state.chart.options.scales.y.min = 0;
            state.chart.options.scales.y.max = 1;
            state.chart.update();
        }
        
        elements.chartOverlay.style.display = 'flex';
        resetPhases();
        elements.resultsTableBody.innerHTML = '';
        elements.resultsCount.textContent = '0 solutions trouvées';
    }

    function updateUIForStart() {
        elements.runBtn.disabled = true;
        elements.runBtn.innerHTML = '<span class="btn-icon">⏳</span>En cours...';
        elements.stopBtn.disabled = false;
        elements.progressFill.style.width = '0%';
        elements.progressText.textContent = '0%';
        elements.progressFill.parentElement.parentElement.style.display = 'block';
        elements.downloadBtn.style.display = 'none';
    }

    function connectSSE() {
        if (state.eventSource) state.eventSource.close();
        
        state.eventSource = new EventSource('/events');
        
        state.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleEvent(data);
            } catch (e) {
                console.error('Erreur parsing:', e);
            }
        };
        
        state.eventSource.onerror = (error) => {
            console.error('SSE error:', error);
            if (state.isOptimizing) {
                updateStatus('🔌', 'Connexion perdue', 'error');
                resetOptimization();
            }
        };
    }

    function handleEvent(data) {
        switch(data.type) {
            case 'phase':
                updatePhase(data.data.phase);
                break;
            case 'generation_start':
                startGeneration(data.data);
                break;
            case 'point':
                addPoint(data.data);
                break;
            case 'generation_end':
                endGeneration(data.data);
                break;
            case 'results':
                showResults(data.data);
                break;
            case 'done':
                optimizationComplete();
                break;
        }
    }

    function startGeneration(data) {
        state.currentGen = data.generation;
        state.totalGen = data.total;
        
        // Mettre à jour la progression
        const progress = (state.currentGen / state.totalGen) * 100;
        elements.progressFill.style.width = `${progress}%`;
        elements.progressText.textContent = `${Math.round(progress)}%`;
        
        // Afficher le label
        const labelText = state.currentGen === 0 ? 'start' : `Génération ${state.currentGen}`;
        showGenerationLabel(labelText);
        
        updateStatus('🔄', `Génération ${state.currentGen}/${state.totalGen} en cours...`, 'info');
    }

    function addPoint(point) {
        if (!state.chart) return;
        
        // Masquer l'overlay après le premier point
        if (state.currentGen >= 0) {
            elements.chartOverlay.style.display = 'none';
        }
        
        // Stocker le point pour ajuster les axes
        state.allPoints.push({
            x: point.training_time,
            y: point.accuracy
        });
        
        // Trouver ou créer le dataset pour cette génération
        let datasetIndex = findOrCreateDataset(state.currentGen);
        
        // Ajouter le point
        state.chart.data.datasets[datasetIndex].data.push({
            x: point.training_time,
            y: point.accuracy
        });
        
        // Ajuster automatiquement les axes
        adjustAxes();
        
        // Mettre à jour le graphique
        state.chart.update();
    }

    function findOrCreateDataset(generation) {
        // Chercher le dataset existant
        for (let i = 0; i < state.chart.data.datasets.length; i++) {
            if (state.chart.data.datasets[i].label === `Génération ${generation}`) {
                return i;
            }
        }
        
        // Créer un nouveau dataset
        const colors = [
            'rgba(96, 165, 250, 0.6)',
            'rgba(139, 92, 246, 0.6)',
            'rgba(16, 185, 129, 0.6)',
            'rgba(245, 158, 11, 0.6)',
            'rgba(239, 68, 68, 0.6)',
            'rgba(168, 85, 247, 0.6)',
            'rgba(6, 182, 212, 0.6)',
            'rgba(251, 191, 36, 0.6)',
        ];
        
        const colorIndex = generation % colors.length;
        const label = generation === 0 ? 'start' : `Génération ${generation}`;
        
        state.chart.data.datasets.push({
            label: label,
            data: [],
            backgroundColor: colors[colorIndex],
            borderColor: colors[colorIndex].replace('0.6', '1'),
            pointRadius: 4,
            pointHoverRadius: 7,
            pointStyle: 'circle',
            showLine: false
        });
        
        return state.chart.data.datasets.length - 1;
    }

    function adjustAxes() {
        if (!state.chart || state.allPoints.length === 0) return;
        
        // Récupérer tous les points
        const xValues = state.allPoints.map(p => p.x);
        const yValues = state.allPoints.map(p => p.y);
        
        // Calculer min/max avec marges
        const xMin = Math.max(0, Math.min(...xValues) * 0.95);
        const xMax = Math.max(...xValues) * 1.05;
        const yMin = Math.max(0, Math.min(...yValues) * 0.95);
        const yMax = Math.min(1, Math.max(...yValues) * 1.05);
        
        // Appliquer les ajustements
        state.chart.options.scales.x.min = xMin;
        state.chart.options.scales.x.max = xMax;
        state.chart.options.scales.y.min = yMin;
        state.chart.options.scales.y.max = yMax;
        
        state.chart.update();
    }

    function endGeneration(data) {
        updateStatus('✅', `Génération ${data.generation} terminée`, 'success');
    }

    function showResults(results) {
        state.paretoData = results.pareto;
        
        setTimeout(() => {
            if (!state.chart) return;
            
            // EFFACER TOUS LES POINTS PRÉCÉDENTS
            state.chart.data.datasets = [];
            
            // Afficher SEULEMENT le front de Pareto
            if (results.pareto && results.pareto.length > 0) {
                const sortedPareto = [...results.pareto].sort((a, b) => a.training_time - b.training_time);
                
                state.chart.data.datasets.push({
                    label: 'Front de Pareto',
                    data: sortedPareto.map(p => ({
                        x: p.training_time,
                        y: p.accuracy
                    })),
                    backgroundColor: 'rgba(239, 68, 68, 0.9)',
                    borderColor: 'rgba(239, 68, 68, 1)',
                    pointRadius: 8,
                    pointHoverRadius: 12,
                    pointStyle: 'circle',
                    showLine: true,
                    fill: false,
                    tension: 0.1,
                    borderWidth: 3
                });
                
                // Ajuster les axes pour le front de Pareto seulement
                adjustAxesForPareto(results.pareto);
            }
            
            // Mettre à jour le tableau
            updateResultsTable(results.pareto);
            
            // Mettre à jour le compteur
            elements.resultsCount.textContent = `${results.pareto.length} solutions Pareto trouvées`;
            
            // Activer le bouton de téléchargement
            elements.downloadBtn.style.display = 'flex';
            
            updateStatus('🏆', 'Front de Pareto final affiché', 'success');
        }, 500);
    }

    function adjustAxesForPareto(paretoData) {
        if (!state.chart || paretoData.length === 0) return;
        
        const xValues = paretoData.map(p => p.training_time);
        const yValues = paretoData.map(p => p.accuracy);
        
        // Calculer min/max avec bonnes marges pour le front de Pareto
        const xMin = Math.max(0, Math.min(...xValues) * 0.9);
        const xMax = Math.max(...xValues) * 1.1;
        const yMin = Math.max(0, Math.min(...yValues) * 0.9);
        const yMax = Math.min(1, Math.max(...yValues) * 1.1);
        
        // S'assurer que les axes montrent une plage visible
        if (yMax - yMin < 0.1) {
            const center = (yMin + yMax) / 2;
            state.chart.options.scales.y.min = Math.max(0, center - 0.05);
            state.chart.options.scales.y.max = Math.min(1, center + 0.05);
        } else {
            state.chart.options.scales.y.min = yMin;
            state.chart.options.scales.y.max = yMax;
        }
        
        if (xMax - xMin < 0.1) {
            const center = (xMin + xMax) / 2;
            state.chart.options.scales.x.min = Math.max(0, center - 0.05);
            state.chart.options.scales.x.max = center + 0.05;
        } else {
            state.chart.options.scales.x.min = xMin;
            state.chart.options.scales.x.max = xMax;
        }
        
        state.chart.update();
    }

    function optimizationComplete() {
        updateStatus('🎉', 'Optimisation terminée avec succès !', 'success');
        resetOptimization();
        if (state.eventSource) state.eventSource.close();
    }

    function resetOptimization() {
        state.isOptimizing = false;
        elements.runBtn.disabled = false;
        elements.runBtn.innerHTML = '<span class="btn-icon">🚀</span>Lancer l\'Optimisation';
        elements.stopBtn.disabled = true;
        elements.progressFill.parentElement.parentElement.style.display = 'none';
    }

    function resetPhases() {
        elements.phases.forEach(phase => phase.classList.remove('active'));
        elements.phaseLines.forEach(line => line.classList.remove('active'));
    }

    function updatePhase(phaseName) {
        const phaseEl = document.getElementById(`phase-${phaseName}`);
        if (phaseEl) {
            phaseEl.classList.add('active');
            
            // Activer la ligne précédente
            const phaseNum = parseInt(phaseEl.dataset.phase || '0');
            if (phaseNum > 1) {
                const lineIndex = phaseNum - 2;
                if (elements.phaseLines[lineIndex]) {
                    elements.phaseLines[lineIndex].classList.add('active');
                }
            }
        }
    }

    function initChart() {
        const ctx = document.getElementById('paretoChart').getContext('2d');
        
        if (state.chart) state.chart.destroy();
        
        state.chart = new Chart(ctx, {
            type: 'scatter',
            data: { datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 0
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Temps d\'entraînement (secondes)',
                            color: '#94a3b8',
                            font: { size: 13, weight: '600' },
                            padding: { top: 10, bottom: 10 }
                        },
                        grid: {
                            color: 'rgba(148, 163, 184, 0.1)',
                            drawBorder: true,
                            borderColor: 'rgba(148, 163, 184, 0.2)'
                        },
                        ticks: {
                            color: '#94a3b8',
                            font: { size: 11 },
                            callback: function(value) {
                                return value.toFixed(2) + 's';
                            }
                        },
                        min: 0,
                        max: 10
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Précision (Accuracy)',
                            color: '#94a3b8',
                            font: { size: 13, weight: '600' },
                            padding: { top: 10, bottom: 10 }
                        },
                        grid: {
                            color: 'rgba(148, 163, 184, 0.1)',
                            drawBorder: true,
                            borderColor: 'rgba(148, 163, 184, 0.2)'
                        },
                        ticks: {
                            color: '#94a3b8',
                            font: { size: 11 },
                            callback: function(value) {
                                return (value * 100).toFixed(1) + '%';
                            }
                        },
                        min: 0,
                        max: 1
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(30, 41, 59, 0.95)',
                        titleColor: '#f8fafc',
                        bodyColor: '#94a3b8',
                        borderColor: '#334155',
                        borderWidth: 1,
                        cornerRadius: 8,
                        padding: 12,
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const point = context.parsed;
                                return `${label}: Précision ${(point.y * 100).toFixed(2)}%, Temps ${point.x.toFixed(3)}s`;
                            }
                        }
                    }
                }
            }
        });
    }

    function showGenerationLabel(text) {
        const oldLabel = document.getElementById('gen-label');
        if (oldLabel) oldLabel.remove();
        
        const label = document.createElement('div');
        label.id = 'gen-label';
        label.textContent = text;
        label.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
            color: white;
            padding: 20px 40px;
            border-radius: 12px;
            font-size: 2.5rem;
            font-weight: 700;
            z-index: 1000;
            animation: fadeInOut 2s ease;
            border: 3px solid #6366f1;
            box-shadow: 0 0 30px rgba(99, 102, 241, 0.5);
            backdrop-filter: blur(10px);
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        `;
        
        document.querySelector('.chart-container').appendChild(label);
        
        setTimeout(() => {
            if (label.parentNode) label.parentNode.removeChild(label);
        }, 1500);
    }

    function updateResultsTable(data) {
        elements.resultsTableBody.innerHTML = '';
        
        if (!data || data.length === 0) return;
        
        data.sort((a, b) => b.accuracy - a.accuracy)
           .slice(0, 15)
           .forEach((item, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td style="color: #10b981; font-weight: bold; font-size: 1.1rem;">
                    ${(item.accuracy * 100).toFixed(2)}%
                </td>
                <td>${item.training_time.toFixed(3)}s</td>
                <td>${item.n_estimators}</td>
                <td>${item.max_depth}</td>
                <td>${item.min_samples_split}</td>
                <td>${item.max_features.toFixed(3)}</td>
            `;
            elements.resultsTableBody.appendChild(row);
        });
    }

    function updateStatus(icon, text, type = 'info') {
        elements.statusIcon.textContent = icon;
        elements.statusText.textContent = text;
        
        elements.status.className = 'status-message';
        if (type) elements.status.classList.add(type);
        
        const colors = {
            info: '#3b82f6',
            success: '#10b981',
            warning: '#f59e0b',
            error: '#ef4444'
        };
        
        elements.status.style.borderLeftColor = colors[type] || colors.info;
    }

    function handleDownload() {
        if (!state.paretoData.length) return;
        
        const headers = ['Précision', 'Temps(s)', 'n_estimators', 'max_depth', 'min_samples_split', 'max_features'];
        const rows = state.paretoData.map(item => [
            item.accuracy.toFixed(6),
            item.training_time.toFixed(6),
            item.n_estimators,
            item.max_depth,
            item.min_samples_split,
            item.max_features.toFixed(6)
        ]);
        
        const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `pareto_front_${new Date().toISOString().slice(0,10)}.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
});