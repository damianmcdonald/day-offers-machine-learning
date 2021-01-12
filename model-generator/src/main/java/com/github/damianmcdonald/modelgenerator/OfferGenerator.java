package com.github.damianmcdonald.modelgenerator;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

import static java.lang.Double.*;

@Service
public class OfferGenerator {

    private static final double ONE_HUNDRED_PERCENT = 100;
    private static final double INCREMENT_BY_DOUBLE = 2;
    private static final double INCREMENT_BY_TRIPLE = 3;

    /* Values for offer creation distribution */
    @Value("${ofertas.totales}")
    private double totalOffers;

    @Value("${ofertas.con.reglas.porcentaje}")
    private double offersWithRulesPercentage;

    @Value("${ofertas.sin.reglas.porcentaje}")
    private double offersWithoutRulesPercentage;

    @Value("${ofertas.con.ruido.porcentaje}")
    private double offersWithNoisePercentage;

    /* Values for min and max offer creation per LINEAL offer type and phase */
    @Value("${model.linear.phase1.min}")
    private double modelLinearPhase1Min;

    @Value("${model.linear.phase1.max}")
    private double modelLinearPhase1Max;

    @Value("${model.linear.phase2.min}")
    private double modelLinearPhase2Min;

    @Value("${model.linear.phase2.max}")
    private double modelLinearPhase2Max;

    @Value("${model.linear.phase3.min}")
    private double modelLinearPhase3Min;

    @Value("${model.linear.phase3.max}")
    private double modelLinearPhase3Max;

    @Value("${model.linear.phase4.porcentaje}")
    private double modelLinearPhase4Percentage;


    /* Values for min and max offer creation per NON LINEAL offer type and phase */
    @Value("${model.nonlinear.phase1.min}")
    private double modelNonLinearPhase1Min;

    @Value("${model.nonlinear.phase1.max}")
    private double modelNonLinearPhase1Max;

    @Value("${model.nonlinear.phase2.min}")
    private double modelNonLinearPhase2Min;

    @Value("${model.nonlinear.phase2.max}")
    private double modelNonLinearPhase2Max;

    @Value("${model.nonlinear.phase3.min}")
    private double modelNonLinearPhase3Min;

    @Value("${model.nonlinear.phase3.max}")
    private double modelNonLinearPhase3Max;

    @Value("${model.nonlinear.phase4.porcentaje}")
    private double modelNonLinearPhase4Percentage;


    /* Values for min and max offer creation per noisy offer type and phase */
    @Value("${model.ruido.phase1.min}")
    private double modelNoisyPhase1Min;

    @Value("${model.ruido.phase1.max}")
    private double modelNoisyPhase1Max;

    @Value("${model.ruido.phase1.incrementor}")
    private double modelNoisyPhase1Incrementor;

    @Value("${model.ruido.phase2.min}")
    private double modelNoisyPhase2Min;

    @Value("${model.ruido.phase2.max}")
    private double modelNoisyPhase2Max;

    @Value("${model.ruido.phase2.incrementor}")
    private double modelNoisyPhase2Incrementor;

    @Value("${model.ruido.phase3.min}")
    private double modelNoisyPhase3Min;

    @Value("${model.ruido.phase3.max}")
    private double modelNoisyPhase3Max;

    @Value("${model.ruido.phase3.incrementor}")
    private double modelNoisyPhase3Incrementor;

    @Value("${model.ruido.phase4.porcentaje}")
    private double modelNoisyPhase4Percentage;

    public static Set<Offer> OFFERS_SET = new HashSet<Offer>();

    @Autowired
    private ComplexityCalculator complexityCalculator;

    @Autowired
    private PhasePredictor phasePredictor;

    public Set<Offer> generateModel() {
        System.out.println(totalOffers + ", " + offersWithRulesPercentage + ", " + offersWithoutRulesPercentage + ", " + offersWithNoisePercentage);
        final long rangeLinear = new Double(totalOffers * (offersWithRulesPercentage / ONE_HUNDRED_PERCENT)).longValue();
        final long rangePhase = new Double(totalOffers * (offersWithoutRulesPercentage / ONE_HUNDRED_PERCENT)).longValue();
        final long rangeNoise = new Double(totalOffers * (offersWithNoisePercentage / ONE_HUNDRED_PERCENT)).longValue();
        System.out.println(rangeLinear + ", " + rangePhase + ", " + rangeNoise);
        Set<Offer> offersLinear = generateModelLinear(rangeLinear);
        Set<Offer> offersPhase = generateModelPhase(rangePhase);
        Set<Offer> offersNoise = generateModelNoise(rangeNoise);
        return new HashSet<Offer>() {{
            addAll(offersLinear);
            addAll(offersPhase);
            addAll(offersNoise);
        } };
    }

    private Set<Offer> generateModelLinear(final long range) {
        final Set<Offer> offers = new HashSet<Offer>();
        for (long i = 0; i < range; i++) {

            final OfferDto offerDto = generateOfferDto();

            final double phase1Prediction = phasePredictor.predictPhase(
                    complexityCalculator.calculateBaseComplexity(offerDto, Phase.RECOPILACION),
                    modelLinearPhase1Min,
                    modelLinearPhase1Max,
                    (offerDto.getAdministered() == 1) ? true : false,
                    Phase.RECOPILACION
            );

            final double phase2Prediction = phasePredictor.predictPhase(
                    complexityCalculator.calculateBaseComplexity(offerDto, Phase.DISENO),
                    (phase1Prediction <= modelLinearPhase1Min) ? modelLinearPhase2Min : phase1Prediction,
                    modelLinearPhase2Max,
                    (offerDto.getAdministered() == 1) ? true : false,
                    Phase.DISENO
            );

            final double phase3Prediction = phasePredictor.predictPhase(
                    complexityCalculator.calculateBaseComplexity(offerDto, Phase.IMPLANTACION),
                    (phase2Prediction <= modelLinearPhase2Min) ? modelLinearPhase3Min : phase2Prediction,
                    modelLinearPhase3Max,
                    (offerDto.getAdministered() == 1) ? true : false,
                    Phase.IMPLANTACION
            );

            final double phase4Prediction = (phase1Prediction + phase2Prediction + phase3Prediction) * modelLinearPhase4Percentage;

            final Offer offer = new Offer.OfferBuilder()
                    .offerDto(offerDto)
                    .phase1prediction(phase1Prediction)
                    .phase2prediction(phase2Prediction)
                    .phase3prediction(phase3Prediction)
                    .phase4prediction(Math.round(phase4Prediction*4)/4f)
                    .build();

            offers.add(offer);

        }
        return offers;
    }

    private Set<Offer> generateModelPhase(final long range) {
        final Set<Offer> offers = new HashSet<Offer>();
        for (long i = 0; i < range; i++) {

            final OfferDto offerDto = generateOfferDto();

            final double phase1Prediction = phasePredictor.predictPhase(
                    complexityCalculator.calculateBaseComplexity(offerDto, Phase.RECOPILACION),
                    modelNonLinearPhase1Min,
                    modelNonLinearPhase1Max,
                    (offerDto.getAdministered() == 1) ? true : false,
                    Phase.RECOPILACION
            );

            final double phase2Prediction = phasePredictor.predictPhase(
                    complexityCalculator.calculateBaseComplexity(offerDto, Phase.DISENO),
                    modelNonLinearPhase2Min,
                    modelNonLinearPhase2Max,
                    (offerDto.getAdministered() == 1) ? true : false,
                    Phase.DISENO
            );

            final double phase3Prediction = phasePredictor.predictPhase(
                    complexityCalculator.calculateBaseComplexity(offerDto, Phase.IMPLANTACION),
                    modelNonLinearPhase3Min,
                    modelNonLinearPhase3Max,
                    (offerDto.getAdministered() == 1) ? true : false,
                    Phase.IMPLANTACION
            );

            final double phase4Prediction = (phase1Prediction + phase2Prediction + phase3Prediction) * modelNonLinearPhase4Percentage;

            final Offer offer = new Offer.OfferBuilder()
                    .offerDto(offerDto)
                    .phase1prediction(phase1Prediction)
                    .phase2prediction(phase2Prediction)
                    .phase3prediction(phase3Prediction)
                    .phase4prediction(Math.round(phase4Prediction*4)/4f)
                    .build();

            offers.add(offer);

        }
        return offers;
    }

    private Set<Offer> generateModelNoise(final long range) {
        final Set<Offer> offers = new HashSet<Offer>();
        for (long i = 0; i < range; i++) {

            final OfferDto offerDto = generateOfferDto();


            int phase1Rand = getRand(1, 24);
            double phase1MaxNoise = 0;
            double phase1MinNoise = 0;

            if(phase1Rand <= 8) {
                phase1MaxNoise = modelNoisyPhase1Max;
                phase1MinNoise = modelNoisyPhase1Min;
            }

            if(phase1Rand > 8 && phase1Rand <= 16) {
                final double incrementedMax = modelNoisyPhase1Max + (modelNoisyPhase1Max * modelNoisyPhase1Incrementor);
                final double incrementedMin = modelNoisyPhase1Min + (modelNoisyPhase1Min * modelNoisyPhase1Incrementor);
                phase1MaxNoise = Math.round(incrementedMax*4)/4f;
                phase1MinNoise = Math.round(incrementedMin*4)/4f;
            }

            if(phase1Rand > 16 && phase1Rand <= 20) {
                final double incrementedMax = modelNoisyPhase1Max + (modelNoisyPhase1Max * modelNoisyPhase1Incrementor * INCREMENT_BY_DOUBLE);
                final double incrementedMin = modelNoisyPhase1Min + (modelNoisyPhase1Min * modelNoisyPhase1Incrementor * INCREMENT_BY_DOUBLE);
                phase1MaxNoise = Math.round(incrementedMax*4)/4f;
                phase1MinNoise = Math.round(incrementedMin*4)/4f;
            }

            if(phase1Rand > 20) {
                final double incrementedMax = modelNoisyPhase1Max + (modelNoisyPhase1Max * modelNoisyPhase1Incrementor * INCREMENT_BY_TRIPLE);
                final double incrementedMin = modelNoisyPhase1Min + (modelNoisyPhase1Min * modelNoisyPhase1Incrementor * INCREMENT_BY_TRIPLE);
                phase1MaxNoise = Math.round(incrementedMax*4)/4f;
                phase1MinNoise = Math.round(incrementedMin*4)/4f;
            }

            final double phase1Prediction = phasePredictor.predictPhase(
                    complexityCalculator.calculateBaseComplexity(offerDto, Phase.RECOPILACION),
                    phase1MinNoise,
                    phase1MaxNoise,
                    (offerDto.getAdministered() == 1) ? true : false,
                    Phase.RECOPILACION
            );

            int phase2Rand = getRand(1, 24);
            double phase2MaxNoise = 0;
            double phase2MinNoise = 0;

            if(phase2Rand <= 8) {
                phase2MaxNoise = modelNoisyPhase2Max;
                phase2MinNoise = modelNoisyPhase2Min;
            }

            if(phase2Rand > 8 && phase2Rand <= 16) {
                final double incrementedMax = modelNoisyPhase2Max + (modelNoisyPhase2Max * modelNoisyPhase2Incrementor);
                final double incrementedMin = modelNoisyPhase2Min + (modelNoisyPhase2Min * modelNoisyPhase2Incrementor);
                phase2MaxNoise = Math.round(incrementedMax*4)/4f;
                phase2MinNoise = Math.round(incrementedMin*4)/4f;
            }

            if(phase2Rand > 16 && phase2Rand <= 20) {
                final double incrementedMax = modelNoisyPhase2Max + (modelNoisyPhase2Max * modelNoisyPhase2Incrementor * INCREMENT_BY_DOUBLE);
                final double incrementedMin = modelNoisyPhase2Min + (modelNoisyPhase2Min * modelNoisyPhase2Incrementor * INCREMENT_BY_DOUBLE);
                phase2MaxNoise = Math.round(incrementedMax*4)/4f;
                phase2MinNoise = Math.round(incrementedMin*4)/4f;
            }

            if(phase2Rand > 20) {
                final double incrementedMax = modelNoisyPhase2Max + (modelNoisyPhase2Max * modelNoisyPhase2Incrementor * INCREMENT_BY_TRIPLE);
                final double incrementedMin = modelNoisyPhase2Min + (modelNoisyPhase2Min * modelNoisyPhase2Incrementor * INCREMENT_BY_TRIPLE);
                phase2MaxNoise = Math.round(incrementedMax*4)/4f;
                phase2MinNoise = Math.round(incrementedMin*4)/4f;
            }

            final double phase2Prediction = phasePredictor.predictPhase(
                    complexityCalculator.calculateBaseComplexity(offerDto, Phase.DISENO),
                    phase2MinNoise,
                    phase2MaxNoise,
                    (offerDto.getAdministered() == 1) ? true : false,
                    Phase.DISENO
            );


            int phase3Rand = getRand(1, 24);
            double phase3MaxNoise = 0;
            double phase3MinNoise = 0;

            if(phase3Rand <= 8) {
                phase3MaxNoise = modelLinearPhase3Max;
                phase3MinNoise = modelLinearPhase3Min;
            }

            if(phase3Rand > 8 && phase3Rand <= 16) {
                final double incrementedMax = modelNoisyPhase3Max + (modelNoisyPhase3Max * modelNoisyPhase3Incrementor);
                final double incrementedMin = modelNoisyPhase3Min + (modelNoisyPhase3Min * modelNoisyPhase3Incrementor);
                phase3MaxNoise = Math.round(incrementedMax*4)/4f;
                phase3MinNoise = Math.round(incrementedMin*4)/4f;
            }

            if(phase3Rand > 16 && phase3Rand <= 20) {
                final double incrementedMax = modelNoisyPhase3Max + (modelNoisyPhase3Max * modelNoisyPhase3Incrementor * INCREMENT_BY_DOUBLE);
                final double incrementedMin = modelNoisyPhase3Min + (modelNoisyPhase3Min * modelNoisyPhase3Incrementor + INCREMENT_BY_DOUBLE);
                phase3MaxNoise = Math.round(incrementedMax*4)/4f;
                phase3MinNoise = Math.round(incrementedMin*4)/4f;
            }

            if(phase3Rand > 20) {
                final double incrementedMax = modelNoisyPhase3Max + (modelNoisyPhase3Max * modelNoisyPhase3Incrementor * INCREMENT_BY_TRIPLE);
                final double incrementedMin = modelNoisyPhase3Min + (modelNoisyPhase3Min * modelNoisyPhase3Incrementor + INCREMENT_BY_TRIPLE);
                phase3MaxNoise = Math.round(incrementedMax*4)/4f;
                phase3MinNoise = Math.round(incrementedMin*4)/4f;
            }

            final double phase3Prediction = phasePredictor.predictPhase(
                    complexityCalculator.calculateBaseComplexity(offerDto, Phase.IMPLANTACION),
                    phase3MinNoise,
                    phase3MaxNoise,
                    (offerDto.getAdministered() == 1) ? true : false,
                    Phase.IMPLANTACION
            );

            final double phase4Prediction = (phase1Prediction + phase2Prediction + phase3Prediction) * modelNoisyPhase4Percentage;

            final Offer offer = new Offer.OfferBuilder()
                    .offerDto(offerDto)
                    .phase1prediction(phase1Prediction)
                    .phase2prediction(phase2Prediction)
                    .phase3prediction(phase3Prediction)
                    .phase4prediction(Math.round(phase4Prediction*4)/4f)
                    .build();

            offers.add(offer);

        }
        return offers;

    }

    private int getRand(final int min, final int max) {
        return ThreadLocalRandom.current().nextInt(min, max + 1);
    }

    private OfferDto generateOfferDto() {
        final int greenfield = getRand(0,1);
        final double vpc = valueOf(getRand(1, 2)) / valueOf("2");

        double subnets = 0.0;
        final int randSubnet = getRand(1, 100);
        if(vpc == 1) {
            if(randSubnet < 80) {
                subnets = valueOf(getRand(3, 4)) / valueOf("4");
            } else {
                subnets = valueOf(getRand(2, 4)) / valueOf("4");
            }
        } else {
            if(randSubnet < 80) {
                subnets = valueOf(getRand(1, 2)) / valueOf("4");
            } else {
                subnets = valueOf(getRand(1, 4)) / valueOf("4");
            }
        }

        final int connectivity = getRand(0, 1);
        final int peerings = getRand(0,1);
        final int directoryservice = getRand(0,1);
        final double otherservices = getRand(0,1);
        final int advsecurity = getRand(0,1);
        final int advlogging = getRand(0,1);
        final int advmonitoring = getRand(0,1);
        final int advbackup = getRand(0,1);
        final double vms = valueOf(getRand(0, 10)) / valueOf("10");
        final double buckets = valueOf(getRand(0, 2)) / valueOf("2");
        final double databases = valueOf(getRand(0, 2)) / valueOf("2");
        final int elb = getRand(0,1);
        final int autoscripts = getRand(0,1);
        final int administered = getRand(0,1);

        return new OfferDto.OfferDtoBuilder()
                .greenfield(greenfield)
                .vpcs(vpc)
                .subnets(subnets)
                .connectivity(connectivity)
                .peerings(peerings)
                .directoryservice(directoryservice)
                .otherservices(otherservices)
                .advsecurity(advsecurity)
                .advlogging(advlogging)
                .advmonitoring(advmonitoring)
                .advbackup(advbackup)
                .vms(vms)
                .buckets(buckets)
                .databases(databases)
                .elb(elb)
                .autoscripts(autoscripts)
                .administered(administered)
                .build();
    }

}
