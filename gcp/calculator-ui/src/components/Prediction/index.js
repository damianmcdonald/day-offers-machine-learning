import React from 'react';
import Faker from 'faker';
import axios from 'axios';
import firebase from 'firebase/app';
import 'firebase/auth';


class Prediction extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            isSignedIn: false,
            client: "",
            greenfield: "yes",
            numVpcs: 0,
            numSubredes: 0,
            connectivity: "no",
            peerings: "no",
            directoryService: "no",
            advSecurity: "no",
            advLogging: "no",
            advMonitoring: "no",
            advBackup: "no",
            numVms: 0,
            numBuckets: 0,
            numDatabases: 0,
            elb: "no",
            autoScripts: "no",
            otrosServicios: 0,
            administered: "no",
            predictedJson: undefined,
            isPersisted: false,
            isLoading: false
          };
        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
      }
    

    /** HELPER FUNCTIONS */
    getRandomBool(majorityVal, minorityVal, probability) {
        const prob = Math.floor(Math.random() * (100 - 1 + 1)) + 1;
        if (prob <= probability) {
            return majorityVal;
        }
        return minorityVal;
    }


    getRandomInt(min, max) {
        min = Math.ceil(min);
        max = Math.floor(max);
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }


    /** AJAX FUNCTIONS */
    postPrediction(idToken) {
        console.log("Posting prediction to GCP Cloud Function");

        this.setState({isLoading: true});

        const offer = {
            greenfield: this.state.greenfield === "yes" ? 1 : 0,
            vpc: this.state.numVpcs / 2,
            subnets: this.state.numSubredes / 4,
            connectivity: this.state.connectivity === "yes" ? 1 : 0,
            peerings: this.state.peerings === "yes" ? 1 : 0,
            directoryservice: this.state.directoryService === "yes" ? 1 : 0,
            otherservices: this.state.otrosServicios / 5,
            advsecurity: this.state.advSecurity === "yes" ? 1 : 0,
            advlogging: this.state.advLogging === "yes" ? 1 : 0,
            advmonitoring: this.state.advMonitoring === "yes" ? 1 : 0,
            advbackup: this.state.advBackup === "yes" ? 1 : 0,
            vms: this.state.numVms / 10,
            buckets: this.state.numBuckets / 2,
            databases: this.state.numDatabases / 2,
            elb: this.state.elb === "yes" ? 1 : 0,
            autoscripts: this.state.autoScripts === "yes" ? 1 : 0,
            administered: this.state.administered === "yes" ? 1 : 0
        }

        const self = this;

        axios({
            method: 'post',
            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + idToken },
            url: 'https://europe-west1-day-offers-ml-poc.cloudfunctions.net/predict',
            data: { offer }
        })
        .then(function (response) {
            console.log(JSON.stringify(response.data, null, 2));
            const predictionJson = JSON.stringify(response.data, null, 2)
            self.setState({predictedJson: JSON.parse(predictionJson)});
            self.setState({isLoading: false});
        })
        .catch(error => {
            self.setState({isLoading: false});
            console.log(error);
            alert(error.response);
        });
    }


    postPersistence(idToken) {
        console.log("Persisting prediction to GCP Cloud Function");
        this.setState({isLoading: true});

        const phasePredictions = {
            client: this.state.client,
            datetime: new Date().toISOString().substring(0, 10),
            phase1prediction: this.state.predictedJson.phase1prediction,
            phase2prediction: this.state.predictedJson.phase2prediction,
            phase3prediction: this.state.predictedJson.phase3prediction,
            phase4prediction: this.state.predictedJson.phase4prediction,
            totalprediction: this.state.predictedJson.totalprediction
        }

        const self = this;

        axios({
            method: 'post',
            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + idToken },
            url: 'https://europe-west1-day-offers-ml-poc.cloudfunctions.net/persist',
            data: { phasePredictions }
        })
        .then(function (response) {
            console.log(JSON.stringify(response.data, null, 2));
            self.setState({isLoading: false});
            self.setState({isPersisted: true});
        })
        .catch(error => {
            self.setState({isLoading: false});
            self.setState({isPersisted: false});
            console.log(error);
            alert(error.response);
        });
    }

    /** EVENT HANDLER FUNCTIONS */
    handleChange(event) {
        const targetId = event.target.id;
        const targetValue = event.target.value;
        switch(targetId) {
            case "inputClient":
                this.setState({client: targetValue});
            break;
            case "selectGreenfield":
                this.setState({greenfield: targetValue});
            break;
            case "inputVpc":
                this.setState({numVpcs: targetValue});
            break;
            case "inputSubredes":
                this.setState({numSubredes: targetValue});
            break;
            case "selectConnectivity":
                this.setState({connectivity: targetValue});
            break;
            case "selectPeerings":
                this.setState({peerings: targetValue});
            break;
            case "selectDirectoryService":
                this.setState({directoryService: targetValue});
            break;
            case "selectAdvSecurity":
                this.setState({advSecurity: targetValue});
            break;
            case "selectAdvLogging":
                this.setState({advLogging: targetValue});
            break;
            case "selectAdvMonitoring":
                this.setState({advMonitoring: targetValue});
            break;
            case "selectAdvBackup":
                this.setState({advBackup: targetValue});
            break;
            case "inputVms":
                this.setState({numVms: targetValue});
            break;
            case "inputBuckets":
                this.setState({numBuckets: targetValue});
            break;
            case "inputDatabases":
                this.setState({numDatabases: targetValue});
            break;
            case "selectElb":
                this.setState({elb: targetValue});
            break;
            case "selectAutoScripts":
                this.setState({autoScripts: targetValue});
            break;
            case "inputOtrosServicios":
                this.setState({otrosServicios: targetValue});
            break;
            case "selectAdministered":
                this.setState({administered: targetValue});
            break;
            default:
                console.log("Form element not found: " + targetId);
        }
    }
    

    handleSubmit(event) {
        const self = this;
        firebase.auth().currentUser.getIdToken(true).then(function(idToken) {
            console.log(idToken);
            self.postPrediction(idToken);
        }).catch(function(error) {
            console.log(error);
            alert(error.response);
        });
        event.preventDefault();
    }


    handlePreFill(self) {
        console.log("Do pre fill");
        self.setState({client: Faker.company.companyName()});
        self.setState({greenfield: self.getRandomBool(1,0,80) === 1 ? "yes" : "no"});
        self.setState({numVpcs: self.getRandomInt(1,2)});
        self.setState({numSubredes: self.getRandomInt(1,4)});
        self.setState({connectivity: self.getRandomBool(0,1,70) === 1 ? "yes" : "no"});
        self.setState({peerings: self.getRandomBool(0,1,90) === 1 ? "yes" : "no"});
        self.setState({directoryService: self.getRandomBool(0,1,80) === 1 ? "yes" : "no"});
        self.setState({advSecurity: self.getRandomBool(0,1,80) === 1 ? "yes" : "no"});
        self.setState({advLogging: self.getRandomBool(0,1,80) === 1 ? "yes" : "no"});
        self.setState({advMoniotoring: self.getRandomBool(0,1,80) === 1 ? "yes" : "no"});
        self.setState({advBackup: self.getRandomBool(0,1,80) === 1 ? "yes" : "no"});
        self.setState({numVms: self.getRandomInt(0,10)});
        self.setState({numBuckets: self.getRandomInt(0,2)});
        self.setState({numDatabases: self.getRandomInt(0,2)});
        self.setState({elb: self.getRandomBool(0,1,70) === 1 ? "yes" : "no"});
        self.setState({autoScripts: self.getRandomBool(0,1,70) === 1 ? "yes" : "no"});
        self.setState({otrosServicios: self.getRandomInt(0,5)});
        self.setState({administered: self.getRandomBool(0,1,70) === 1 ? "yes" : "no"});
        self.setState({predictedJson: undefined});

        // reset the dynamic elements
        self.setState({isLoading: false});
        self.setState({isPersisted: false});
    }

    handleReset(self) {
        console.log("Do reset");
        self.setState({client: Faker.company.companyName()});
        self.setState({greenfield: "no"});
        self.setState({numVpcs: 0});
        self.setState({numSubredes: 0});
        self.setState({connectivity: "no"});
        self.setState({peerings: "no"});
        self.setState({directoryService: "no"});
        self.setState({advSecurity: "no"});
        self.setState({advLogging: "no"});
        self.setState({advMoniotoring: "no"});
        self.setState({advBackup: "no"});
        self.setState({numVms: 0});
        self.setState({numBuckets: 0});
        self.setState({numDatabases: 0});
        self.setState({elb: "no"});
        self.setState({autoScripts: "no"});
        self.setState({otrosServicios: 0});
        self.setState({administered: "no"});

        // reset previous predictions
        self.setState({predictedJson: undefined});

        // reset the dynamic elements
        self.setState({isLoading: false});
        self.setState({isPersisted: false});
    }

    handlePersist(self) {
        console.log("Do persist");
        self.setState({isLoading: true});
        firebase.auth().currentUser.getIdToken(true).then(function(idToken) {
            console.log(idToken);
            self.postPersistence(idToken);
        }).catch(function(error) {
            self.setState({isLoading: false});
            self.setState({isPersisted: false});
            console.log(error);
            alert(error.response);
        });
    }

    
    /** REACT RENDER FUNCTION */
    render() {
        return (
            <div className="row"> 
                <div className="col-7">
                    <div>
                        <h1>Calculadora</h1>
                    </div>
                    <form onSubmit={this.handleSubmit}>
                        <div className="form-row">
                        <div className="col">
                                <label>Nombre del cliente</label>
                                <br/>
                                <input type="text" value={this.state.client} onChange={this.handleChange} id="inputClient" placeholder="Nombre del cliente" />
                            </div>
                        </div>
                        <hr/>
                        <h4>Tenancy</h4>
                        <div className="form-row">
                            <div className="col">
                                <label>Proyecto greenfield:</label>
                                <br/>
                                <select value={this.state.greenfield} onChange={this.handleChange} id="selectGreenfield">                           
                                    <option value="yes">Sí</option>
                                    <option value="no">No</option>
                                </select>
                            </div>
                        </div>
                        <hr/>
                        <h4>Networking</h4>
                        <div className="form-row">
                            <div className="col">
                                <label># VPC</label>
                                <br/>
                                <input type="text" className="form-control mb-2 mr-sm-2" value={this.state.numVpcs} onChange={this.handleChange} id="inputVpc" placeholder="Numero de VPCs (1 -2)" />
                            </div>
                            <div className="col">
                                <label># Subredes</label>
                                <br/>
                                <input type="text" className="form-control mb-2 mr-sm-2" value={this.state.numSubredes} onChange={this.handleChange} id="inputSubredes" placeholder="Numero de Subredes (1 -4)" />
                            </div>
                            <div className="col">
                                <label>Conectividad VPN:</label>
                                    <br/>
                                    <select className="form-control mb-2 mr-sm-2" value={this.state.connectivity} onChange={this.handleChange} id="selectConnectivity">
                                        <option value="yes">Sí</option>
                                        <option value="no">No</option>
                                    </select>
                            </div>
                            <div className="col">
                                <label>VPC Peerings:</label>
                                <br/>
                                <select className="form-control mb-2 mr-sm-2" value={this.state.peerings} onChange={this.handleChange} id="selectPeerings">
                                <option value="yes">Sí</option>
                                    <option value="no">No</option>
                                </select>
                            </div>
                        </div>
                        <hr/>
                        <h4>Security</h4>
                        <div className="form-row">
                            <div className="col">
                                <label>Directory Service (MSAD o LDAP):</label>
                                <br/>
                                <select value={this.state.directoryService} onChange={this.handleChange} id="selectDirectoryService">
                                <option value="yes">Sí</option>
                                    <option value="no">No</option>
                                </select>
                            </div>
                            <div className="col">
                                <label>Seguridad advanzada:</label>
                                <br/>
                                <select value={this.state.advSecurity} onChange={this.handleChange} id="selectAdvSecurity">
                                    <option value="yes">Sí</option>
                                    <option value="no">No</option>
                                </select>
                            </div>
                        </div>
                        <hr/>
                        <h4>Monitoring and Management</h4>
                        <div className="form-row">
                            <div className="col">
                                <label>Logging advanzado:</label>
                                <br/>
                                <select className="form-control mb-2 mr-sm-2" value={this.state.advLogging} onChange={this.handleChange} id="selectAdvLogging">
                                    <option value="yes">Sí</option>
                                    <option value="no">No</option>
                                </select>
                            </div>
                            <div className="col">
                                <label>Monitoring advanzado:</label>
                                <br/>
                                <select className="form-control mb-2 mr-sm-2" value={this.state.advMoniotoring} onChange={this.handleChange} id="selectAdvMonitoring">
                                    <option value="yes">Sí</option>
                                    <option value="no">No</option>
                                </select>
                            </div>
                            <div className="col">
                                <label>Backup advanzado:</label>
                                <br/>
                                <select className="form-control mb-2 mr-sm-2" value={this.state.advBackup} onChange={this.handleChange} id="selectAdvBackup">
                                    <option value="yes">Sí</option>
                                    <option value="no">No</option>
                                </select>
                            </div>
                        </div>
                        <hr/>
                        <h4>Deployment</h4>
                        <div className="form-row">
                            <div className="col">
                                <label># Máquinas Virtuales</label>
                                <br/>
                                <input type="text" className="form-control mb-2 mr-sm-2" value={this.state.numVms} onChange={this.handleChange} id="inputVms" placeholder="Numero de Máquinas Virtuales (0 -10)" />
                            </div>
                            <div className="col">
                                <label># Buckets</label>
                                <br/>
                                <input type="text" className="form-control mb-2 mr-sm-2" value={this.state.numBuckets} onChange={this.handleChange} id="inputBuckets" placeholder="Numero de Buckets (0 -2)" />
                            </div>
                            <div className="col">
                                <label># BBDD</label>
                                <br/>
                                <input type="text" className="form-control mb-2 mr-sm-2" value={this.state.numDatabases} onChange={this.handleChange} id="inputDatabases" placeholder="Numero de BBDD (0 -2)" />
                            </div>
                        </div>
                        <br/>
                        <div className="form-row">
                            <div className="col">
                                <label>Load Balancer:</label>
                                <br/>
                                <select value={this.state.elb} onChange={this.handleChange} id="selectElb">
                                    <option value="yes">Sí</option>
                                    <option value="no">No</option>
                                </select>
                            </div>
                            <div className="col">
                                <label>Scripts de automatizacion:</label>
                                <br/>
                                <select value={this.state.autoScripts} onChange={this.handleChange} id="selectAutoScripts">
                                    <option value="yes">Sí</option>
                                    <option value="no">No</option>
                                </select>
                            </div>
                        </div>
                        <hr/>
                        <h4>Otros servicios</h4>
                        <div className="form-row">
                            <div className="col">
                                    <label># otros servicios</label>
                                    <br/>
                                    <input type="text" value={this.state.otrosServicios} onChange={this.handleChange} id="inputOtrosServicios" placeholder="Numero de otros servicios (0 -5)" />
                            </div>
                            <div className="col">
                                <label>¿Es administrado?:</label>
                                <br/>
                                <select value={this.state.administered} onChange={this.handleChange} id="selectAdministered">
                                    <option value="yes">Sí</option>
                                    <option value="no">No</option>
                                </select>
                            </div>
                        </div>

                        <hr/>
                        <div className="form-row">
                            <div className="col">
                                <button type="button" className="btn btn-primary" onClick={() => {this.handlePreFill(this);}}>Pre-Fill</button>
                            </div>
                            <div className="col">
                                <button type="button" className="btn btn-warning" onClick={() => {this.handleReset(this);}}>Reset</button>
                            </div>
                            <div className="col">
                                <button type="submit" className="btn btn-success" disabled={this.state.isLoading}>Submit</button>
                            </div>
                            <div className="col">
                                <button type="button" className="btn btn-dark" disabled={this.state.isLoading} onClick={() => {this.handlePersist(this);}}>Persist</button>
                            </div>
                            <div className="col">
                                <button type="button" className="btn btn-danger" disabled={this.state.isLoading} onClick={() => firebase.auth().signOut()}>Sign Out</button>
                            </div>
                        </div>
                    </form>
                </div>
                <div className="col-3">
                    <div>
                        <h1>Predicciones</h1>
                    </div>
                    {this.state.isLoading && 
                        <div><img src="spinner.gif" alt="Spinner graphic" /></div>
                    }
                    {!this.state.isLoading && 
                    <div>
                        {(this.state.predictedJson !== undefined) &&
                        
                        <table className="table table-striped">
                            <thead className="thead-dark">
                                <tr>
                                <th scope="col">Fase</th>
                                <th scope="col">Predicción</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                <td>Recopilación</td>
                                <td>{this.state.predictedJson.phase1prediction}</td>
                                </tr>
                                <tr>
                                <td>Diseño</td>
                                <td>{this.state.predictedJson.phase2prediction}</td>
                                </tr>
                                <tr>
                                <td>Implantación</td>
                                <td>{this.state.predictedJson.phase3prediction}</td>
                                </tr>
                                <tr>
                                <td>Soporte</td>
                                <td>{this.state.predictedJson.phase4prediction}</td>
                                </tr>
                                <tr>
                                <td>Total</td>
                                <td>{this.state.predictedJson.totalprediction}</td>
                                </tr>
                            </tbody>
                        </table>
                        }
                    </div>
                    }
                    {(!this.state.isLoading && this.state.isPersisted) &&
                        <div>
                          {(this.state.predictedJson !== undefined) &&  
                          <div>
                              <p>Details persisted for client:</p>
                              <p><strong>{(this.state.client)}</strong></p>
                              <img src="smiley-face.png" alt="Smiley Face" />
                            </div>
                            }
                            </div>
                        }
                </div>
            </div>
        );
    }
}

export default Prediction;