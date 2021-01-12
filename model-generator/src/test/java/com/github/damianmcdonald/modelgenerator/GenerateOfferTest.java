package com.github.damianmcdonald.modelgenerator;

import com.github.damianmcdonald.modelgenerator.CsvGenerator;
import com.github.damianmcdonald.modelgenerator.Offer;
import com.github.damianmcdonald.modelgenerator.OfferDto;
import com.github.damianmcdonald.modelgenerator.OfferGenerator;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

import java.io.IOException;
import java.util.Set;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

@RunWith(SpringRunner.class)
@SpringBootTest
public class GenerateOfferTest {

    @Autowired
    CsvGenerator csvGenerator;

    @Autowired
    OfferGenerator offerGenerator;


    @Test
    public void createCSVFileTest() throws IOException {
        final Set<Offer> offers = offerGenerator.generateModel();
        csvGenerator.createCSVFile(offers);
    }

    @Test
    public void verifyOfferEqualityTest() {

        final int TRUE_INT = 1;
        final int FALSE_INT = 0;
        final double BASE_VAL = 0.5;
        final double PHASE_1_PREDICTION = 1.5;
        final double PHASE_2_PREDICTION = 2.0;
        final double PHASE_3_PREDICTION = 4.0;
        final double PHASE_4_PREDICTION = 1.25;

        final OfferDto offerDtoA = new OfferDto.OfferDtoBuilder()
                .greenfield(TRUE_INT)
                .vpcs(BASE_VAL)
                .subnets(BASE_VAL)
                .connectivity(TRUE_INT)
                .peerings(TRUE_INT)
                .directoryservice(TRUE_INT)
                .otherservices(TRUE_INT)
                .advsecurity(FALSE_INT)
                .advlogging(FALSE_INT)
                .advmonitoring(FALSE_INT)
                .advbackup(FALSE_INT)
                .vms(BASE_VAL)
                .buckets(BASE_VAL)
                .databases(BASE_VAL)
                .elb(TRUE_INT)
                .autoscripts(TRUE_INT)
                .administered(TRUE_INT)
                .build();

        final Offer offerA = new Offer.OfferBuilder()
                .offerDto(offerDtoA)
                .phase1prediction(PHASE_1_PREDICTION)
                .phase2prediction(PHASE_2_PREDICTION)
                .phase3prediction(PHASE_3_PREDICTION)
                .phase4prediction(PHASE_4_PREDICTION)
                .build();

        final OfferDto offerDtoB = new OfferDto.OfferDtoBuilder()
                .greenfield(TRUE_INT)
                .vpcs(BASE_VAL)
                .subnets(BASE_VAL)
                .connectivity(TRUE_INT)
                .peerings(TRUE_INT)
                .directoryservice(TRUE_INT)
                .otherservices(TRUE_INT)
                .advsecurity(FALSE_INT)
                .advlogging(FALSE_INT)
                .advmonitoring(FALSE_INT)
                .advbackup(FALSE_INT)
                .vms(BASE_VAL)
                .buckets(BASE_VAL)
                .databases(BASE_VAL)
                .elb(TRUE_INT)
                .autoscripts(TRUE_INT)
                .administered(TRUE_INT)
                .build();

        final Offer offerB = new Offer.OfferBuilder()
                .offerDto(offerDtoB)
                .phase1prediction(PHASE_1_PREDICTION)
                .phase2prediction(PHASE_2_PREDICTION)
                .phase3prediction(PHASE_3_PREDICTION)
                .phase4prediction(PHASE_4_PREDICTION)
                .build();

        final boolean isEqual = offerA.equals(offerB);
        System.out.println("OfferA and OfferB is equal - expect true == " + isEqual);
        assertTrue(isEqual);
    }

    @Test
    public void verifyOfferInEqualityTest() {

        final int TRUE_INT = 1;
        final int FALSE_INT = 0;
        final double BASE_VAL = 0.5;
        final double PHASE_1_PREDICTION = 1.5;
        final double PHASE_2_PREDICTION = 2.0;
        final double PHASE_3_PREDICTION = 4.0;
        final double PHASE_4_PREDICTION = 1.25;

        final OfferDto offerDtoA = new OfferDto.OfferDtoBuilder()
                .greenfield(TRUE_INT)
                .vpcs(BASE_VAL)
                .subnets(BASE_VAL)
                .connectivity(TRUE_INT)
                .peerings(TRUE_INT)
                .directoryservice(TRUE_INT)
                .otherservices(TRUE_INT)
                .advsecurity(FALSE_INT)
                .advlogging(FALSE_INT)
                .advmonitoring(FALSE_INT)
                .advbackup(FALSE_INT)
                .vms(BASE_VAL)
                .buckets(BASE_VAL)
                .databases(BASE_VAL)
                .elb(TRUE_INT)
                .autoscripts(TRUE_INT)
                .administered(TRUE_INT)
                .build();

        final Offer offerA = new Offer.OfferBuilder()
                .offerDto(offerDtoA)
                .phase1prediction(PHASE_1_PREDICTION)
                .phase2prediction(PHASE_2_PREDICTION)
                .phase3prediction(PHASE_3_PREDICTION)
                .phase4prediction(PHASE_4_PREDICTION)
                .build();

        final OfferDto offerDtoB = new OfferDto.OfferDtoBuilder()
                .greenfield(FALSE_INT)
                .vpcs(BASE_VAL)
                .subnets(BASE_VAL)
                .connectivity(TRUE_INT)
                .peerings(TRUE_INT)
                .directoryservice(TRUE_INT)
                .otherservices(TRUE_INT)
                .advsecurity(FALSE_INT)
                .advlogging(FALSE_INT)
                .advmonitoring(FALSE_INT)
                .advbackup(FALSE_INT)
                .vms(BASE_VAL)
                .buckets(BASE_VAL)
                .databases(BASE_VAL)
                .elb(TRUE_INT)
                .autoscripts(TRUE_INT)
                .administered(TRUE_INT)
                .build();

        final Offer offerB = new Offer.OfferBuilder()
                .offerDto(offerDtoB)
                .phase1prediction(PHASE_1_PREDICTION)
                .phase2prediction(PHASE_2_PREDICTION)
                .phase3prediction(PHASE_3_PREDICTION)
                .phase4prediction(PHASE_4_PREDICTION)
                .build();

        final boolean isEqual = offerA.equals(offerB);
        System.out.println("OfferA and OfferB note equal - expect false == " + isEqual);
        assertFalse(isEqual);
    }
}
