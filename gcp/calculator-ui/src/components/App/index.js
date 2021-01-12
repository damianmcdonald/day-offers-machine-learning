import React from 'react';
import { Container, Row } from 'react-bootstrap';
import firebase from 'firebase/app';
import 'firebase/auth';
import StyledFirebaseAuth from 'react-firebaseui/StyledFirebaseAuth';
import Prediction from '../Prediction'
import Header from '../Header'
import Footer from '../Footer'

// Get the Firebase config from the auto generated file.
const firebaseConfig = require('../../firebase-config.json').result;

// Instantiate a Firebase app.
const firebaseApp = firebase.initializeApp(firebaseConfig);


class App extends React.Component {
  state = {
    isSignedIn: undefined
  };

  uiConfig = {
    signInFlow: 'popup',
    signInOptions: [
      firebase.auth.GoogleAuthProvider.PROVIDER_ID,
      firebase.auth.EmailAuthProvider.PROVIDER_ID
    ],
    callbacks: {
      signInSuccessWithAuthResult: () => false
    },
  };

  componentDidMount() {
    this.unregisterAuthObserver = firebaseApp.auth().onAuthStateChanged((user) => {
      this.setState({isSignedIn: !!user});
    });
  }

  componentWillUnmount() {
    this.unregisterAuthObserver();
  }

  render() {
    return (
      <div>
      <Header></Header>
      <br/>
      <Container>
        {this.state.isSignedIn !== undefined && !this.state.isSignedIn &&
          <Row>
            <StyledFirebaseAuth uiConfig={this.uiConfig}
                                firebaseAuth={firebaseApp.auth()}/>
          </Row>
        }
        {this.state.isSignedIn && 
          <Prediction isSignedIn={this.state.isSignedIn}>
          </Prediction>
        }
      </Container>
      <hr/>
      <Footer></Footer>
      </div>
    );
  }
}

export default App;