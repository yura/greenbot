class AddLatLonToRecyclers < ActiveRecord::Migration[7.0]
  def change
    add_column :recyclers, :lat, :decimal, precision: 8, scale: 6
    add_column :recyclers, :lon, :decimal, precision: 9, scale: 6
  end
end
