<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use App\Models\DaftarPelatihan;
use App\Models\User;

class Pelatihan extends Model
{
    protected $guarded = ['id'];

    public function user()
    {
        return $this->belongsTo(User::class);
    }

    public function daftarPelatihan()
    {
        return $this->hasMany(DaftarPelatihan::class);
    }

    public function mitra()
    {
        return $this->belongsTo(User::class, 'mitra_id');
    }
}
