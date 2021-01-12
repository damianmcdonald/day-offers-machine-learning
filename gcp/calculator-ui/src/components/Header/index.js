import React from 'react';

class Header extends React.Component {

  render() {
    return (
      <div className="card card-image" style={{ backgroundImage: 'url(head-foot.jpg)', height: 200 }}>
        <div className="text-white text-center py-5 px-4 my-5">
          <div>
            <h2 className="card-title h1-responsive pt-3 mb-5 font-bold"><strong>Aprendizaje automático</strong></h2>
          </div>
        </div>
      </div>
    );
  }

}

export default Header;