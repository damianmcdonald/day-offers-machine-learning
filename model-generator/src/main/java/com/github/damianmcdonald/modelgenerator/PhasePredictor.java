package com.github.damianmcdonald.modelgenerator;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class PhasePredictor {

    @Autowired
    private ComplexityCalculator complexityCalculator;

    public double predictPhase(final double complexity, final double phaseMin, final double phaseMax, final boolean isAdministered, final Phase phase) {
        final int phaseSpecifcComplexity =
                Phase.IMPLANTACION.equals(phase) ? complexityCalculator.getMaxComplexity(isAdministered) : complexityCalculator.getMaxComplexity(false);
        final double conversionFactor = phaseMax / phaseSpecifcComplexity;
        final double phaseEffort = complexity * conversionFactor;
        final double phasePrediction = Math.round(phaseEffort*4)/4f;
        return phasePrediction > phaseMin ? phasePrediction : phaseMin;
    }
}
